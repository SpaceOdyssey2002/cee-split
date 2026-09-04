using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Unity.RenderStreaming;
using System;

public class DynamicRenderNode : MonoBehaviour
{
    [Header("节点身份")]
    public int myNodeID = 0;

    [Header("渲染控制")]
    public Camera renderCamera;
    public VideoStreamSender videoSender;
    public Vector2Int baseResolution = new Vector2Int(1920, 1080);

    [Header("场景监控 (挂载即可)")]
    public AutoAdaptiveTriCounter triCounter;

    private UdpClient udpListener; // 收指令 (9999)
    private Thread listenThread;
    private bool isRunning = true; // 🌟 安全退出标识

    private string latestCommand = "";
    private readonly object lockObj = new object();
    private float currentScale = 1.0f;
    private int currentMask = -1;
    private int currentBitrateKbps = 0;
    private SenderFramePacker framePacker;

    private static readonly float[] ResolutionScales = { 0.33f, 0.67f, 1.0f, 1.33f, 2.0f };
    private static readonly Vector2Int[] ResolutionProfiles =
    {
        new Vector2Int(640, 360),
        new Vector2Int(1280, 720),
        new Vector2Int(1920, 1080),
        new Vector2Int(2560, 1440),
        new Vector2Int(3840, 2160)
    };

    void Awake()
    {
        Application.runInBackground = true; // 🌟 保证后台运行，防止编辑器失焦卡死
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "-node" && i + 1 < args.Length)
            {
                if (int.TryParse(args[i + 1], out int id))
                {
                    myNodeID = id;
                    Debug.Log($"[节点初始化] 获取身份: Node {myNodeID}");
                }
            }
        }

        // Keep the telemetry packet identity aligned with the runtime node id.
        // AutoAdaptiveTriCounter is the single owner of node telemetry on port 9998.
        if (triCounter != null)
            triCounter.myNodeID = myNodeID;
    }

    void Start()
    {
        framePacker = renderCamera != null ? renderCamera.GetComponent<SenderFramePacker>() : null;
        if (framePacker != null && framePacker.videoStreamSender == null)
        {
            framePacker.videoStreamSender = videoSender;
        }

        // Apply the initial 1080p profile even before the first scheduler packet.
        // The packed source is fixed at the maximum size; WebRTC only changes its
        // encoder downscale factor, so no RTP track replacement is required.
        if (!ChangeResolution(currentScale))
            currentScale = -1.0f;

        // 1. 初始化接收指令的 UDP (9999)
        udpListener = new UdpClient();
        udpListener.ExclusiveAddressUse = false;
        udpListener.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
        udpListener.Client.Bind(new System.Net.IPEndPoint(System.Net.IPAddress.Any, 9999));

        listenThread = new Thread(ListenUDP) { IsBackground = true };
        listenThread.Start();

        // AutoAdaptiveTriCounter sends one complete node telemetry packet
        // (triangles, render/encode time and sender BWE) directly to the phone.
        // Do not send a second LAN-broadcast packet from this component.
    }

    void ListenUDP()
    {
        System.Net.IPEndPoint groupEP = new System.Net.IPEndPoint(System.Net.IPAddress.Any, 9999);
        while (isRunning)
        {
            try
            {
                byte[] bytes = udpListener.Receive(ref groupEP);
                lock (lockObj)
                {
                    latestCommand = Encoding.UTF8.GetString(bytes);
                }
            }
            catch (SocketException)
            {
                // 忽略非阻塞或关闭时的异常
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[ListenUDP] 接收异常: {ex.Message}");
            }
        }
    }

    void Update()
    {
        string cmd;
        lock (lockObj)
        {
            if (string.IsNullOrEmpty(latestCommand)) return;
            cmd = latestCommand;
            latestCommand = ""; // 取出后清空，防止重复执行
        }

        try
        {
            ProcessCommand(cmd);
        }
        catch (Exception ex)
        {
            Debug.LogError($"[ProcessCommand] 执行异常: {ex.Message}");
        }
    }

    void ProcessCommand(string json)
    {
        RLResponse response = JsonUtility.FromJson<RLResponse>(json);
        if (response == null || response.near == null || response.mid == null || response.far == null) return;

        int newMask = 0;
        float newScale = 0.16f;
        bool renderNear = response.near.node == myNodeID;
        bool renderMid = response.mid.node == myNodeID;
        bool renderFar = response.far.node == myNodeID;
        Vector2Int nearResolution = ResolutionForScale(response.near.scale);
        Vector2Int midResolution = ResolutionForScale(response.mid.scale);
        Vector2Int farResolution = ResolutionForScale(response.far.scale);

        // Rendering resolution remains a per-layer decision. Network rate is
        // aggregated because all layers assigned to this node share one stream.
        int newBitrate = 500;

        // 检查分配给我的层级
        if (renderNear)
        {
            newMask |= (1 << LayerMask.NameToLayer("RenderNear"));
            newScale = Mathf.Max(newScale, response.near.scale);
            newBitrate = Mathf.Max(newBitrate, response.near.bitrate);
        }
        if (renderMid)
        {
            newMask |= (1 << LayerMask.NameToLayer("RenderMid"));
            newScale = Mathf.Max(newScale, response.mid.scale);
            newBitrate = Mathf.Max(newBitrate, response.mid.bitrate);
        }
        if (renderFar)
        {
            newMask |= (1 << LayerMask.NameToLayer("RenderFar"));
            newScale = Mathf.Max(newScale, response.far.scale);
            newBitrate = Mathf.Max(newBitrate, response.far.bitrate);
        }

        // 1. Render every assigned layer with its own resolution. SenderFramePacker
        // scales the resulting color/depth pairs to the largest node canvas.
        if (framePacker != null)
        {
            framePacker.ConfigureLayers(
                renderNear, nearResolution,
                renderMid, midResolution,
                renderFar, farResolution);
            renderCamera.enabled = false;
            currentMask = newMask;
        }
        else if (currentMask != newMask)
        {
            renderCamera.cullingMask = newMask;
            currentMask = newMask;
            renderCamera.enabled = newMask != 0;

            if (renderFar)
            {
                renderCamera.clearFlags = CameraClearFlags.Skybox;
            }
            else
            {
                renderCamera.clearFlags = CameraClearFlags.SolidColor;
                renderCamera.backgroundColor = new Color(0, 0, 0, 0f);
            }
        }

        // 2. Apply the rendering-resolution decision independently.
        if (Mathf.Abs(currentScale - newScale) > 0.01f && newMask != 0)
        {
            if (ChangeResolution(newScale))
                currentScale = newScale;
        }

        // 3. Apply the WebRTC target-rate decision independently.
        if (Mathf.Abs(currentBitrateKbps - newBitrate) > 50 && newMask != 0)
        {
            if (ChangeWebRTCBitrate(newBitrate))
                currentBitrateKbps = newBitrate;
        }
    }

    bool ChangeResolution(float scale)
    {
        int profileIndex = FindNearestResolutionProfile(scale);
        Vector2Int requestedColorResolution = ResolutionProfiles[profileIndex];

        Vector2Int appliedColorResolution = framePacker != null &&
            framePacker.CurrentColorResolution.x > 0
            ? framePacker.CurrentColorResolution
            : requestedColorResolution;

        if (videoSender == null)
            return false;

        Vector2Int maximumResolution = framePacker != null
            ? framePacker.MaximumColorResolution
            : ResolutionProfiles[ResolutionProfiles.Length - 1];
        float downscaleFactor = maximumResolution.x / (float)appliedColorResolution.x;

        try
        {
            videoSender.SetScaleResolutionDown(Mathf.Max(1f, downscaleFactor));
            Debug.Log(
                $"[ChangeResolution] profile={profileIndex}, requestedColor=" +
                $"{requestedColorResolution.x}x{requestedColorResolution.y}, appliedColor=" +
                $"{appliedColorResolution.x}x{appliedColorResolution.y}, " +
                $"packedSource={maximumResolution.x}x{maximumResolution.y * 2}, " +
                $"encodedPacked={appliedColorResolution.x}x{appliedColorResolution.y * 2}, " +
                $"encoderDownscale={downscaleFactor:F2}.");
            return true;
        }
        catch (InvalidOperationException)
        {
            // The encoder is not connected yet. The next decision will retry.
            return false;
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[ChangeResolution] WebRTC downscale failed: {ex.Message}");
            return false;
        }
    }

    private static int FindNearestResolutionProfile(float scale)
    {
        int bestIndex = 0;
        float bestDistance = Mathf.Abs(scale - ResolutionScales[0]);
        for (int i = 1; i < ResolutionScales.Length; i++)
        {
            float distance = Mathf.Abs(scale - ResolutionScales[i]);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    private static Vector2Int ResolutionForScale(float scale)
    {
        return ResolutionProfiles[FindNearestResolutionProfile(scale)];
    }

    bool ChangeWebRTCBitrate(int targetKbps)
    {
        if (videoSender == null) return false;
        uint minKbps = (uint)(targetKbps * 0.5f);
        uint maxKbps = (uint)targetKbps;

        try
        {
            videoSender.SetBitrate(minKbps, maxKbps);
            Debug.Log($"[ChangeWebRTCBitrate] target={targetKbps} kbps, range={minKbps}-{maxKbps} kbps.");
            return true;
        }
        catch (InvalidOperationException)
        {
            // WebRTC encoder is not ready. Keep the old value so the next
            // scheduler update retries the same independent rate decision.
            return false;
        }
        catch (Exception)
        {
            return false;
        }
    }

    void OnDestroy()
    {
        isRunning = false; // 🌟 通知线程退出

        udpListener?.Close();
        if (listenThread != null && listenThread.IsAlive)
        {
            listenThread.Join(500); // 最多等 500ms，防止卡死
        }

    }
}

// ==========================================
// 数据结构类定义 (补全渲染节点缺失的解析结构)
// ==========================================
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
