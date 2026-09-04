using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class RLBrainClient : MonoBehaviour
{
    [Header("Python RL Server")]
    public string serverIP = "127.0.0.1";
    public int serverPort = 8080;

    [Header("Nodes Config")]
    public int phoneNodeID = 2;

    [Header("References")]
    public DepthCompositor compositor;
    public RenderStreamingMonitor[] webrtcMonitors;

    [Header("References - 按节点分组")]
    public RenderStreamingMonitor monitorPhoneLocal;
    public RenderStreamingMonitor monitorPhoneRecv1;
    public RenderStreamingMonitor monitorPhoneRecv2;

    private TcpClient client;
    private NetworkStream stream;
    private Thread receiveThread;

    private UdpClient udpBroadcaster;
    private UdpClient udpStatsListener;
    private Thread statsListenThread;

    private string latestCommandJson = "";
    private readonly object lockObj = new object();
    private bool isRunning = true;

    [Header("渲染节点 IPs")]
    public string[] nodeIPs = { "127.0.0.1", "100.109.194.55", "100.113.34.83" };

    private int trisNear = 0, trisMid = 0, trisFar = 0;
    private float currentPowerMw = 0f;
    private Dictionary<int, double> remoteEncodeTimes = new Dictionary<int, double>();
    private Dictionary<int, double> remoteFrameTimes = new Dictionary<int, double>();
    private Dictionary<int, double> remoteBandwidthMbps = new Dictionary<int, double>();
    private Dictionary<int, bool> remoteBandwidthValid = new Dictionary<int, bool>();
    private Dictionary<int, DateTime> remoteBandwidthUpdatedAt = new Dictionary<int, DateTime>();
    private readonly object statsLock = new object();
    private const double RemoteBandwidthMaxAgeSeconds = 3.0;

    void Awake()
    {
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = 1200;
        Application.runInBackground = true;
        Debug.Log("🚀 已解除移动端 30 FPS 锁定，并开启后台运行保护！");
    }

    void Start()
    {
        udpBroadcaster = new UdpClient { EnableBroadcast = true };

        udpStatsListener = new UdpClient();
        udpStatsListener.ExclusiveAddressUse = false;
        udpStatsListener.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
        udpStatsListener.Client.Bind(new IPEndPoint(IPAddress.Any, 9998));

        statsListenThread = new Thread(ListenStatsUDP) { IsBackground = true };
        statsListenThread.Start();

        ConnectToPython();
        InvokeRepeating(nameof(UpdateBatteryStats), 1f, 1f);

        StartCoroutine(SendInitialState());
    }

    IEnumerator SendInitialState()
    {
        yield return new WaitForSeconds(1.0f);
        RequestDecisionAndSend();
    }

    void ListenStatsUDP()
    {
        IPEndPoint groupEP = new IPEndPoint(IPAddress.Any, 9998);
        while (isRunning)
        {
            try
            {
                byte[] bytes = udpStatsListener.Receive(ref groupEP);
                string json = Encoding.UTF8.GetString(bytes);
                NodeStats stats = JsonUtility.FromJson<NodeStats>(json);

                lock (statsLock)
                {
                    trisNear = stats.n; trisMid = stats.m; trisFar = stats.f;
                    remoteEncodeTimes[stats.node] = stats.enc;
                    remoteFrameTimes[stats.node] = stats.rft;
                    remoteBandwidthMbps[stats.node] = stats.bw;
                    remoteBandwidthValid[stats.node] = stats.bw_valid != 0 &&
                        stats.bw > 0.0 && !double.IsNaN(stats.bw) && !double.IsInfinity(stats.bw);
                    remoteBandwidthUpdatedAt[stats.node] = DateTime.UtcNow;
                }
            }
            catch { break; }
        }
    }

    void UpdateBatteryStats() { /* 保持原逻辑不变 */ }

    void ConnectToPython()
    {
        try
        {
            client = new TcpClient(serverIP, serverPort);
            stream = client.GetStream();
            receiveThread = new Thread(ReceiveData) { IsBackground = true };
            receiveThread.Start();
            Debug.Log("✅[RL Brain] 已连接 Python");
        }
        catch (Exception e) { Debug.LogError($"❌[RL Brain] 连接失败: {e.Message}"); }
    }

    void RequestDecisionAndSend()
    {
        if (client == null || !client.Connected) return;

        double c0r = 0.0, c0e = 0.0;
        double c1r = 0.0, c1e = 0.0;
        double senderBw0 = 0.0, senderBw1 = 0.0;
        bool senderBw0Valid = false, senderBw1Valid = false;

        lock (statsLock)
        {
            if (remoteFrameTimes.ContainsKey(0)) c0r = remoteFrameTimes[0];
            if (remoteEncodeTimes.ContainsKey(0)) c0e = remoteEncodeTimes[0];

            if (remoteFrameTimes.ContainsKey(1)) c1r = remoteFrameTimes[1];
            if (remoteEncodeTimes.ContainsKey(1)) c1e = remoteEncodeTimes[1];

            DateTime now = DateTime.UtcNow;
            if (remoteBandwidthMbps.ContainsKey(0) &&
                remoteBandwidthValid.TryGetValue(0, out bool valid0) && valid0 &&
                remoteBandwidthUpdatedAt.TryGetValue(0, out DateTime updated0) &&
                (now - updated0).TotalSeconds <= RemoteBandwidthMaxAgeSeconds)
            {
                senderBw0 = remoteBandwidthMbps[0];
                senderBw0Valid = true;
            }
            if (remoteBandwidthMbps.ContainsKey(1) &&
                remoteBandwidthValid.TryGetValue(1, out bool valid1) && valid1 &&
                remoteBandwidthUpdatedAt.TryGetValue(1, out DateTime updated1) &&
                (now - updated1).TotalSeconds <= RemoteBandwidthMaxAgeSeconds)
            {
                senderBw1 = remoteBandwidthMbps[1];
                senderBw1Valid = true;
            }
        }

        double pr = 0.0;
        double prCurrent = 0.0;
        int prValid = 0;
        int localActive = 0;
        string prSource = "none";
        RenderStreamingMonitor localMon = monitorPhoneLocal;
        if (localMon == null && webrtcMonitors != null)
        {
            localMon = webrtcMonitors.FirstOrDefault(m => m != null && m.isLocalNode);
        }
        if (localMon != null)
        {
            pr = localMon.GetLocalGpuLatencyEstimateMs();
            prCurrent = localMon.GetLocalGpuRenderTimeMs();
            prValid = localMon.HasLocalGpuTiming() ? 1 : 0;
            localActive = localMon.IsLocalCameraRendering() ? 1 : 0;
            prSource = localMon.GetLocalGpuTimingSourceName();
        }

        // ✅ 新增：t0/bw0 为云端传输耗时与带宽, t1/bw1 为边缘端传输耗时与带宽
        double d0 = 0.0, r0 = 0.0, j0 = 0.0, t0 = 0.0, bw0 = 0.0, rx0 = 0.0;
        double d1 = 0.0, r1 = 0.0, j1 = 0.0, t1 = 0.0, bw1 = 0.0, rx1 = 0.0;
        int bw0Valid = 0, bw1Valid = 0;
        RenderStreamingMonitor mon0 = null, mon1 = null;

        if (webrtcMonitors != null)
        {
            mon0 = webrtcMonitors.FirstOrDefault(m => m != null && !m.isLocalNode && m.nodeID == 0 && m.IsMonitoring());
            if (mon0 != null)
            {
                d0 = mon0.GetDecodeTime();
                r0 = mon0.GetRTT();
                j0 = mon0.GetJitter();
                rx0 = mon0.GetBitrate() / 1000.0;
            }

            mon1 = webrtcMonitors.FirstOrDefault(m => m != null && !m.isLocalNode && m.nodeID == 1 && m.IsMonitoring());
            if (mon1 != null)
            {
                d1 = mon1.GetDecodeTime();
                r1 = mon1.GetRTT();
                j1 = mon1.GetJitter();
                rx1 = mon1.GetBitrate() / 1000.0;
            }
        }

        // 优先采用发送节点的发送侧 BWE。若该 UDP 样本刚好过期，
        // 则使用接收 monitor 已经验证过的统一带宽值。这个统一值既可以
        // 来自 WebRTC incoming BWE，也可以来自上一次转发给 monitor 的 sender BWE。
        if (senderBw0Valid)
        {
            bw0 = senderBw0;
            bw0Valid = 1;
        }
        else if (mon0 != null && mon0.HasAvailableBandwidth())
        {
            bw0 = mon0.GetAvailableBandwidth();
            bw0Valid = 1;
        }
        if (senderBw1Valid)
        {
            bw1 = senderBw1;
            bw1Valid = 1;
        }
        else if (mon1 != null && mon1.HasAvailableBandwidth())
        {
            bw1 = mon1.GetAvailableBandwidth();
            bw1Valid = 1;
        }

        if (mon0 != null && bw0Valid != 0)
            t0 = mon0.EstimateTransmissionDelayMs(bw0);
        if (mon1 != null && bw1Valid != 0)
            t1 = mon1.EstimateTransmissionDelayMs(bw1);

        int n, m, f;
        lock (statsLock) { n = trisNear; m = trisMid; f = trisFar; }

        // ✅ JSON 字段对齐（请确保 Python 端的 state 解析逻辑同步加上 t0, bw0, t1, bw1）
        string json = "{" +
            "\"n\":" + n + ",\"m\":" + m + ",\"f\":" + f + "," +
            "\"bw0\":" + bw0.ToString("F3", CultureInfo.InvariantCulture) + "," +
            "\"bw0_valid\":" + bw0Valid + "," +
            "\"rx0\":" + rx0.ToString("F3", CultureInfo.InvariantCulture) + "," +
            "\"t0\":" + t0.ToString("F1", CultureInfo.InvariantCulture) + "," +
            "\"bw1\":" + bw1.ToString("F3", CultureInfo.InvariantCulture) + "," +
            "\"bw1_valid\":" + bw1Valid + "," +
            "\"rx1\":" + rx1.ToString("F3", CultureInfo.InvariantCulture) + "," +
            "\"t1\":" + t1.ToString("F1", CultureInfo.InvariantCulture) + "," +
            "\"c0r\":" + c0r.ToString("F1", CultureInfo.InvariantCulture) + "," +
            "\"c0e\":" + c0e.ToString("F1", CultureInfo.InvariantCulture) + "," +
            "\"c1r\":" + c1r.ToString("F1", CultureInfo.InvariantCulture) + "," +
            "\"c1e\":" + c1e.ToString("F1", CultureInfo.InvariantCulture) + "," +
            "\"pr\":" + pr.ToString("F1", CultureInfo.InvariantCulture) + "," +
            "\"pr_current\":" + prCurrent.ToString("F1", CultureInfo.InvariantCulture) + "," +
            "\"pr_valid\":" + prValid + "," +
            "\"local_active\":" + localActive + "," +
            "\"pr_source\":\"" + prSource + "\"," +
            "\"d0\":" + d0.ToString("F1", CultureInfo.InvariantCulture) + "," +
            "\"r0\":" + r0.ToString("F1", CultureInfo.InvariantCulture) + "," +
            "\"j0\":" + j0.ToString("F1", CultureInfo.InvariantCulture) + "," +
            "\"d1\":" + d1.ToString("F1", CultureInfo.InvariantCulture) + "," +
            "\"r1\":" + r1.ToString("F1", CultureInfo.InvariantCulture) + "," +
            "\"j1\":" + j1.ToString("F1", CultureInfo.InvariantCulture) +
            "}\n";

        try
        {
            byte[] data = Encoding.UTF8.GetBytes(json);
            stream.Write(data, 0, data.Length);
        }
        catch (Exception ex) { Debug.LogWarning($"[RL] 数据发送失败: {ex.Message}"); }
    }

    void ReceiveData()
    {
        byte[] buffer = new byte[4096];
        StringBuilder sb = new StringBuilder();

        while (isRunning && client != null && client.Connected)
        {
            try
            {
                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                if (bytesRead > 0)
                {
                    string msg = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    sb.Append(msg);

                    if (msg.Contains("\n"))
                    {
                        string[] commands = sb.ToString().Split('\n');
                        lock (lockObj) { latestCommandJson = commands[commands.Length - 2]; }
                        sb.Clear();
                        if (!msg.EndsWith("\n")) sb.Append(commands[commands.Length - 1]);
                    }
                }
            }
            catch { break; }
        }
    }

    void Update()
    {
        ApplySenderBandwidthToReceiverMonitors();

        string currentJson;
        lock (lockObj)
        {
            if (string.IsNullOrEmpty(latestCommandJson)) return;
            currentJson = latestCommandJson;
            latestCommandJson = "";
        }

        try
        {
            RLResponse response = JsonUtility.FromJson<RLResponse>(currentJson);
            if (response == null || compositor == null) return;

            compositor.UpdateLayerMapping(response.near.node, response.mid.node, response.far.node);

            byte[] cmdBytes = Encoding.UTF8.GetBytes(currentJson);
            if (udpBroadcaster != null && nodeIPs != null)
            {
                foreach (string ip in nodeIPs) udpBroadcaster.Send(cmdBytes, cmdBytes.Length, ip, 9999);
            }

            StartCoroutine(WaitAndReplyState());
        }
        catch (Exception ex) { Debug.LogError($"[RL Brain] Update 异常: {ex.Message}"); }
    }

    private void ApplySenderBandwidthToReceiverMonitors()
    {
        if (webrtcMonitors == null)
            return;

        double bandwidth0 = 0.0;
        double bandwidth1 = 0.0;
        bool valid0 = false;
        bool valid1 = false;

        lock (statsLock)
        {
            DateTime now = DateTime.UtcNow;
            if (remoteBandwidthMbps.TryGetValue(0, out double value0) &&
                remoteBandwidthValid.TryGetValue(0, out bool sampleValid0) &&
                sampleValid0 &&
                remoteBandwidthUpdatedAt.TryGetValue(0, out DateTime updated0) &&
                (now - updated0).TotalSeconds <= RemoteBandwidthMaxAgeSeconds)
            {
                bandwidth0 = value0;
                valid0 = true;
            }

            if (remoteBandwidthMbps.TryGetValue(1, out double value1) &&
                remoteBandwidthValid.TryGetValue(1, out bool sampleValid1) &&
                sampleValid1 &&
                remoteBandwidthUpdatedAt.TryGetValue(1, out DateTime updated1) &&
                (now - updated1).TotalSeconds <= RemoteBandwidthMaxAgeSeconds)
            {
                bandwidth1 = value1;
                valid1 = true;
            }
        }

        foreach (RenderStreamingMonitor monitor in webrtcMonitors)
        {
            if (monitor == null || monitor.isLocalNode)
                continue;

            if (monitor.nodeID == 0)
                monitor.SetExternalAvailableBandwidth(bandwidth0, valid0);
            else if (monitor.nodeID == 1)
                monitor.SetExternalAvailableBandwidth(bandwidth1, valid1);
        }
    }

    IEnumerator WaitAndReplyState()
    {
        yield return new WaitForSeconds(0.2f);
        RequestDecisionAndSend();
    }

    void OnDestroy()
    {
        isRunning = false;

        stream?.Close();
        client?.Close();

        udpBroadcaster?.Close();
        udpStatsListener?.Close();

        if (receiveThread != null && receiveThread.IsAlive) receiveThread.Join(500);
        if (statsListenThread != null && statsListenThread.IsAlive) statsListenThread.Join(500);
    }

    [Serializable]
    public class RLResponse
    {
        public LayerDecision near;
        public LayerDecision mid;
        public LayerDecision far;
    }

    [Serializable]
    public class LayerDecision
    {
        public int node;
        public float scale;
        public int bitrate;
    }

    [Serializable]
    public class NodeStats
    {
        public int n, m, f;
        public int node;
        public double enc;
        public double rft;
        public double bw;
        public int bw_valid;
    }
}
