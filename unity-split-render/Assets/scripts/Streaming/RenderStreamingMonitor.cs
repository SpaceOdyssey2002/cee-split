using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using Unity.WebRTC;
using Unity.RenderStreaming;

public class RenderStreamingMonitor : MonoBehaviour
{
    [Header("UI Settings")]
    public Vector2 uiPosition = new Vector2(20, 20);
    public int fontSize = 24;
    public Color textColor = Color.green;
    public bool showUI = true;
    public bool showFPS = true;

    [Header("Node Settings")]
    public int nodeID = 0;
    public bool isLocalNode = false;

    [Header("WebRTC Stats")]
    [Min(0.25f)] public float statsIntervalSeconds = 0.5f;
    [Min(0.5f)] public float bandwidthHoldSeconds = 3.0f;
    [Range(0.01f, 1.0f)] public float bandwidthSmoothing = 0.25f;

    [Header("Local Render Optimization")]
    [Tooltip("仅在 isLocalNode 为 true 时分配，指向挂载了 LocalRenderTimer 的摄像机")]
    public LocalRenderTimer localRenderTimer;

    private SingleConnection singleConnection;
    private SignalingManager signalingManager;
    private RTCPeerConnection peerConnection;

    private string currentConnId = "";
    private bool isMonitoringWebRTC = false;
    private float fps = 0f;
    private float frameTimeMs = 0f;
    private float localCpuSubmissionMs = 0f;
    private float presentationFrameTimeMs = 0f;
    private float deltaTime = 0.0f;
    private GUIStyle _style;

    // 基础网络与编解码数据
    private double rttMs = 0;
    private double jitterMs = 0;
    private long packetsLost = 0;
    private double bitrateKbps = 0;
    private double encodeTimeMs = 0;
    private double decodeTimeMs = 0;

    // 抗抖动缓冲等待时间
    private double jitterBufferDelayMs = 0;
    private double lastJitterBufferDelay = 0;
    private ulong lastJitterBufferEmittedCount = 0;

    // 带宽与传输时延
    private double availableBandwidthMbps = 0;
    private double availableIncomingBandwidthMbps = 0;
    private double availableOutgoingBandwidthMbps = 0;
    private double externalAvailableBandwidthMbps = 0;
    private bool availableBandwidthValid = false;
    private bool incomingBandwidthValid = false;
    private bool outgoingBandwidthValid = false;
    private bool externalBandwidthValid = false;
    private float lastIncomingBandwidthTime = float.NegativeInfinity;
    private float lastOutgoingBandwidthTime = float.NegativeInfinity;
    private float lastExternalBandwidthTime = float.NegativeInfinity;
    private string bandwidthSource = "unavailable";
    private double transmissionDelayMs = 0;

    // 编解码器与码率统计（内部使用）
    private string sendCodec = "N/A";
    private string recvCodec = "N/A";
    private double lastDecodeTime = 0;
    private uint lastFramesDecoded = 0;
    private double lastEncodeTime = 0;
    private uint lastFramesEncoded = 0;
    private ulong lastBytesReceived = 0;
    private ulong lastBytesSent = 0;
    private double sendBitrateKbps = 0;
    private double receiveFps = 0;
    private double averageReceivedFrameBytes = 0;
    private float lastFrameSizeSampleTime = float.NegativeInfinity;
    private string transmissionDelayStatus = "waiting-for-stats";
    private float lastStatTime = 0;

    private void ResetStatsBaselines()
    {
        lastDecodeTime = 0;
        lastFramesDecoded = 0;
        lastEncodeTime = 0;
        lastFramesEncoded = 0;
        lastBytesReceived = 0;
        lastBytesSent = 0;
        lastJitterBufferDelay = 0;
        lastJitterBufferEmittedCount = 0;
        lastStatTime = 0;
        availableBandwidthMbps = 0;
        availableIncomingBandwidthMbps = 0;
        availableOutgoingBandwidthMbps = 0;
        externalAvailableBandwidthMbps = 0;
        availableBandwidthValid = false;
        incomingBandwidthValid = false;
        outgoingBandwidthValid = false;
        externalBandwidthValid = false;
        lastIncomingBandwidthTime = float.NegativeInfinity;
        lastOutgoingBandwidthTime = float.NegativeInfinity;
        lastExternalBandwidthTime = float.NegativeInfinity;
        bandwidthSource = "unavailable";
        transmissionDelayMs = 0;
        receiveFps = 0;
        averageReceivedFrameBytes = 0;
        lastFrameSizeSampleTime = float.NegativeInfinity;
        transmissionDelayStatus = "waiting-for-stats";
    }

    private static bool IsPositiveFinite(double value)
    {
        return value > 0.0 && !double.IsNaN(value) && !double.IsInfinity(value);
    }

    private double SmoothBandwidth(double previousMbps, double sampleMbps)
    {
        if (!IsPositiveFinite(previousMbps))
            return sampleMbps;

        double alpha = Mathf.Clamp01(bandwidthSmoothing);
        return previousMbps + alpha * (sampleMbps - previousMbps);
    }

    void Start()
    {
        _style = new GUIStyle();
        _style.fontSize = fontSize;
        _style.normal.textColor = textColor;
        if (isLocalNode) return;

        singleConnection = GetComponent<SingleConnection>();
        signalingManager = FindObjectOfType<SignalingManager>();
        if (singleConnection != null && signalingManager != null)
        {
            StartCoroutine(AutoHookPeerConnection());
        }
    }

    void Update()
    {
        deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;
        presentationFrameTimeMs = deltaTime * 1000.0f;
        fps = deltaTime > 0.0001f ? 1.0f / deltaTime : 0f;
        if (isLocalNode && localRenderTimer != null)
        {
            frameTimeMs = (float)localRenderTimer.GetGpuRenderTimeMs();
            localCpuSubmissionMs = (float)localRenderTimer.GetCpuSubmissionTimeMs();
        }
        else
        {
            frameTimeMs = deltaTime * 1000.0f;
        }
    }

    IEnumerator AutoHookPeerConnection()
    {
        FieldInfo connIdField = typeof(SingleConnection).GetField(
            "connectionId", BindingFlags.NonPublic | BindingFlags.Instance);
        if (connIdField == null) yield break;

        while (true)
        {
            string connId = null;
            while (string.IsNullOrEmpty(connId))
            {
                if (singleConnection == null) yield break;
                connId = connIdField.GetValue(singleConnection) as string;
                yield return new WaitForSeconds(0.5f);
            }

            currentConnId = connId;
            yield return new WaitForSeconds(1.0f);

            peerConnection = ExtractPeerConnection(signalingManager, connId);

            if (peerConnection != null && IsPeerConnectionUsable(peerConnection))
            {
                ResetStatsBaselines();
                isMonitoringWebRTC = true;
                yield return StartCoroutine(UpdateWebRTCStats());
            }
            else
            {
                yield return new WaitForSeconds(2.0f);
            }

            isMonitoringWebRTC = false;
            peerConnection = null;
            currentConnId = "";
            yield return new WaitForSeconds(1.0f);
        }
    }

    private bool IsPeerConnectionUsable(RTCPeerConnection connection)
    {
        if (connection == null) return false;

        try
        {
            RTCPeerConnectionState state = connection.ConnectionState;
            return state != RTCPeerConnectionState.Closed &&
                   state != RTCPeerConnectionState.Failed;
        }
        catch (System.ObjectDisposedException)
        {
            return false;
        }
        catch (System.InvalidOperationException)
        {
            return false;
        }
    }

    private RTCPeerConnection ExtractPeerConnection(SignalingManager manager, string connId)
    {
        try
        {
            var instanceField = typeof(SignalingManager).GetField(
                "m_instance", BindingFlags.NonPublic | BindingFlags.Instance);
            if (instanceField == null) return null;
            var m_instance = instanceField.GetValue(manager);
            if (m_instance == null) return null;

            foreach (var field in m_instance.GetType().GetFields(
                BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public))
            {
                if (typeof(IDictionary).IsAssignableFrom(field.FieldType))
                {
                    var dict = field.GetValue(m_instance) as IDictionary;
                    if (dict != null && dict.Contains(connId))
                    {
                        var sessionObj = dict[connId];
                        if (sessionObj == null) continue;

                        foreach (var sf in sessionObj.GetType().GetFields(
                            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
                        {
                            if (sf.FieldType == typeof(RTCPeerConnection))
                                return sf.GetValue(sessionObj) as RTCPeerConnection;
                        }
                        foreach (var sp in sessionObj.GetType().GetProperties(
                            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
                        {
                            if (sp.PropertyType == typeof(RTCPeerConnection))
                                return sp.GetValue(sessionObj) as RTCPeerConnection;
                        }
                    }
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError("[Monitor] 反射提取异常：" + e.Message);
        }
        return null;
    }

    private IEnumerator UpdateWebRTCStats()
    {
        while (isMonitoringWebRTC && IsPeerConnectionUsable(peerConnection))
        {
            RTCStatsReportAsyncOperation op;
            try
            {
                op = peerConnection.GetStats();
            }
            catch (System.ObjectDisposedException)
            {
                break;
            }
            catch (System.InvalidOperationException)
            {
                break;
            }

            yield return op;
            if (op.IsError) break;

            RTCStatsReport report = op.Value;
            float currentTime = Time.realtimeSinceStartup;
            float timeDiff = currentTime - lastStatTime;

            var codecMap = new Dictionary<string, string>();
            var selectedCandidatePairIds = new HashSet<string>();
            foreach (var stat in report.Stats.Values)
            {
                if (stat is RTCCodecStats codecStat)
                    codecMap[codecStat.Id] = codecStat.mimeType;
                else if (stat is RTCTransportStats transport &&
                         !string.IsNullOrEmpty(transport.selectedCandidatePairId))
                    selectedCandidatePairIds.Add(transport.selectedCandidatePairId);
            }

            RTCIceCandidatePairStats selectedPair = null;
            RTCIceCandidatePairStats nominatedPair = null;
            foreach (var stat in report.Stats.Values)
            {
                if (!(stat is RTCIceCandidatePairStats pair) || pair.state != "succeeded")
                    continue;

                if (selectedCandidatePairIds.Contains(pair.Id))
                {
                    selectedPair = pair;
                    break;
                }

                if (pair.nominated && nominatedPair == null)
                    nominatedPair = pair;
            }
            selectedPair = selectedPair ?? nominatedPair;

            bool sawInboundVideo = false;
            bool sawOutboundVideo = false;

            foreach (var stat in report.Stats.Values)
            {
                // 接收端统计（手机解码）
                if (stat is RTCInboundRTPStreamStats inbound && inbound.kind == "video")
                {
                    sawInboundVideo = true;
                    jitterMs = inbound.jitter * 1000.0;
                    packetsLost = inbound.packetsLost;

                    uint framesDelta = inbound.framesDecoded >= lastFramesDecoded
                        ? inbound.framesDecoded - lastFramesDecoded : 0;
                    ulong bytesDelta = inbound.bytesReceived >= lastBytesReceived
                        ? inbound.bytesReceived - lastBytesReceived : 0;
                    if (framesDelta > 0)
                    {
                        decodeTimeMs = (inbound.totalDecodeTime - lastDecodeTime) / framesDelta * 1000.0;
                        if (timeDiff > 0 && lastStatTime > 0)
                            receiveFps = framesDelta / timeDiff;

                        // Use bytes and decoded frames from the same WebRTC stats window.
                        // This avoids dividing two independently smoothed/updated metrics.
                        if (bytesDelta > 0 && lastStatTime > 0)
                        {
                            averageReceivedFrameBytes = bytesDelta / (double)framesDelta;
                            lastFrameSizeSampleTime = currentTime;
                        }
                    }
                    lastDecodeTime = inbound.totalDecodeTime;
                    lastFramesDecoded = inbound.framesDecoded;

                    ulong emittedDelta = inbound.jitterBufferEmittedCount >= lastJitterBufferEmittedCount
                        ? inbound.jitterBufferEmittedCount - lastJitterBufferEmittedCount : 0;
                    if (emittedDelta > 0)
                        jitterBufferDelayMs = ((inbound.jitterBufferDelay - lastJitterBufferDelay) / emittedDelta) * 1000.0;
                    lastJitterBufferDelay = inbound.jitterBufferDelay;
                    lastJitterBufferEmittedCount = inbound.jitterBufferEmittedCount;

                    if (timeDiff > 0 && lastStatTime > 0)
                        bitrateKbps = (bytesDelta * 8.0) / (timeDiff * 1000.0);
                    lastBytesReceived = inbound.bytesReceived;

                    if (!string.IsNullOrEmpty(inbound.codecId) && codecMap.TryGetValue(inbound.codecId, out var rc))
                        recvCodec = rc;
                }
                // 发送端统计（仅当该节点也发流时才有值，手机通常为空）
                else if (stat is RTCOutboundRTPStreamStats outbound &&
                         (outbound.kind == "video" || outbound.framesEncoded > 0))
                {
                    sawOutboundVideo = true;
                    uint framesDelta = outbound.framesEncoded >= lastFramesEncoded
                        ? outbound.framesEncoded - lastFramesEncoded : 0;
                    if (framesDelta > 0)
                        encodeTimeMs = (outbound.totalEncodeTime - lastEncodeTime) / framesDelta * 1000.0;
                    lastEncodeTime = outbound.totalEncodeTime;
                    lastFramesEncoded = outbound.framesEncoded;

                    if (timeDiff > 0 && lastStatTime > 0 && outbound.bytesSent >= lastBytesSent)
                        sendBitrateKbps = ((outbound.bytesSent - lastBytesSent) * 8.0) / (timeDiff * 1000.0);
                    lastBytesSent = outbound.bytesSent;

                    if (!string.IsNullOrEmpty(outbound.codecId) && codecMap.TryGetValue(outbound.codecId, out var sc))
                        sendCodec = sc;
                }
            }

            if (selectedPair != null)
            {
                rttMs = selectedPair.currentRoundTripTime * 1000.0;

                double incomingBps = selectedPair.availableIncomingBitrate;
                if (IsPositiveFinite(incomingBps))
                {
                    availableIncomingBandwidthMbps = SmoothBandwidth(
                        availableIncomingBandwidthMbps, incomingBps / 1_000_000.0);
                    lastIncomingBandwidthTime = currentTime;
                }

                double outgoingBps = selectedPair.availableOutgoingBitrate;
                if (IsPositiveFinite(outgoingBps))
                {
                    availableOutgoingBandwidthMbps = SmoothBandwidth(
                        availableOutgoingBandwidthMbps, outgoingBps / 1_000_000.0);
                    lastOutgoingBandwidthTime = currentTime;
                }
            }

            float holdSeconds = Mathf.Max(0.5f, bandwidthHoldSeconds);
            incomingBandwidthValid = currentTime - lastIncomingBandwidthTime <= holdSeconds;
            outgoingBandwidthValid = currentTime - lastOutgoingBandwidthTime <= holdSeconds;
            bool externalBandwidthFresh =
                externalBandwidthValid &&
                currentTime - lastExternalBandwidthTime <= holdSeconds;

            // 容量估计必须和媒体方向一致。实际视频码率仅表示流量需求，不能作为链路容量回退值。
            if (sawInboundVideo && incomingBandwidthValid)
            {
                availableBandwidthMbps = availableIncomingBandwidthMbps;
                availableBandwidthValid = true;
                bandwidthSource = "incoming-bwe";
            }
            else if (sawInboundVideo && externalBandwidthFresh)
            {
                availableBandwidthMbps = externalAvailableBandwidthMbps;
                availableBandwidthValid = true;
                bandwidthSource = "sender-udp-bwe";
            }
            else if (sawOutboundVideo && outgoingBandwidthValid)
            {
                availableBandwidthMbps = availableOutgoingBandwidthMbps;
                availableBandwidthValid = true;
                bandwidthSource = "outgoing-bwe";
            }
            else
            {
                availableBandwidthMbps = 0.0;
                availableBandwidthValid = false;
                bandwidthSource = "unavailable";
            }

            // 传输时延 = 单帧数据量 / 可用带宽
            // 单帧数据量(bits) = bitrateKbps*1000 / fps；带宽(bps) = availableBandwidthMbps*1e6
            if (sawOutboundVideo && !sawInboundVideo)
            {
                transmissionDelayMs = 0.0;
                transmissionDelayStatus = "sender-only: t computed on phone";
            }
            else
            {
                transmissionDelayMs = CalculateTransmissionDelayMs(
                    availableBandwidthMbps, out transmissionDelayStatus);
            }

            lastStatTime = currentTime;
            yield return new WaitForSeconds(Mathf.Max(0.25f, statsIntervalSeconds));
        }

        isMonitoringWebRTC = false;
    }

    // OnGUI：只显示原始采集指标，不显示端到端预估延迟。
    void OnGUI()
    {
        if (!showUI) return;

        GUILayout.BeginArea(new Rect(uiPosition.x, uiPosition.y, Screen.width * 0.6f, Screen.height * 0.8f));
        GUILayout.BeginVertical("box");

        if (showFPS)
            GUILayout.Label($"整体FPS: {fps:F1}", _style);

        if (isLocalNode)
        {
            if (localRenderTimer == null)
                GUILayout.Label($"【Local Node {nodeID}】未配置 LocalRenderTimer", _style);
            else if (!localRenderTimer.IsGpuTimingSupported())
                GUILayout.Label($"【Local Node {nodeID}】GPU计时: 当前设备/图形API不支持", _style);
            else if (!localRenderTimer.IsCameraActivelyRendering())
            {
                GUILayout.Label($"【Local Node {nodeID}】当前未渲染本地场景: 0.00 ms", _style);
                if (localRenderTimer.HasGpuTimingSample())
                    GUILayout.Label($"  最近有效GPU样本: {localRenderTimer.GetLastValidGpuRenderTimeMs():F2} ms", _style);
            }
            else if (!localRenderTimer.HasGpuTimingSample())
                GUILayout.Label($"【Local Node {nodeID}】GPU计时: 等待延迟采样...", _style);
            else if (!localRenderTimer.IsPerCameraGpuTiming())
                GUILayout.Label($"【Local Node {nodeID}】整帧GPU回退: {frameTimeMs:F2} ms", _style);
            else
                GUILayout.Label($"【Local Node {nodeID}】纯本地相机GPU: {frameTimeMs:F2} ms", _style);

            GUILayout.Label($"  相机CPU提交: {localCpuSubmissionMs:F2} ms", _style);
            GUILayout.Label($"  实际帧周期: {presentationFrameTimeMs:F2} ms", _style);
            GUILayout.Label($"  RL本地延迟估计(pr): {GetLocalGpuLatencyEstimateMs():F2} ms", _style);
        }
        else if (isMonitoringWebRTC)
        {
            GUILayout.Label($"━━ Node {nodeID} 原始指标 ━━", _style);
            string bandwidthText = availableBandwidthValid
                ? $"{availableBandwidthMbps:F2} Mbps ({bandwidthSource})"
                : "不可用（等待 WebRTC BWE）";
            GUILayout.Label($"  可用带宽:   {bandwidthText}", _style);
            GUILayout.Label($"  RTT(r):     {rttMs:F1} ms", _style);
            GUILayout.Label($"  传输(t):    {transmissionDelayMs:F1} ms", _style);
            GUILayout.Label($"  t计算状态:  {transmissionDelayStatus}", _style);
            GUILayout.Label($"  解码(d):    {decodeTimeMs:F1} ms", _style);
            GUILayout.Label($"  Jitter(j):  {jitterMs:F1} ms", _style);
            GUILayout.Label($"  接收码率:   {bitrateKbps:F0} kbps", _style);
            GUILayout.Label($"  接收FPS:    {receiveFps:F1}", _style);
            GUILayout.Label($"  平均帧大小: {averageReceivedFrameBytes / 1024.0:F1} KiB", _style);

            // 优先显示接收编解码器。
            string displayCodec = "N/A";
            if (!string.IsNullOrEmpty(recvCodec) && recvCodec != "N/A")
                displayCodec = "Recv: " + recvCodec;
            else if (!string.IsNullOrEmpty(sendCodec) && sendCodec != "N/A")
                displayCodec = "Send: " + sendCodec;

            GUILayout.Label($"  编解码:     {displayCodec}", _style);
            // 端到端延迟由 Python 统一计算。
        }
        else
        {
            GUILayout.Label($"【Node {nodeID}】等待连接...", _style);
        }

        GUILayout.EndVertical();
        GUILayout.EndArea();
    }

    // 对外暴露原始指标，供 RLBrainClient 上报给 Python。
    public bool IsMonitoring() => isLocalNode || isMonitoringWebRTC;
    public float GetFPS() => fps;
    public double GetFrameTimeMs() => frameTimeMs;
    public bool HasLocalGpuTiming() => isLocalNode && localRenderTimer != null && localRenderTimer.HasGpuTimingSample();
    public bool IsLocalCameraRendering() =>
        isLocalNode && localRenderTimer != null && localRenderTimer.IsCameraActivelyRendering();
    public string GetLocalGpuTimingSourceName() =>
        localRenderTimer != null ? localRenderTimer.GetGpuTimingSourceName() : "none";
    public double GetLocalGpuRenderTimeMs() => HasLocalGpuTiming() ? frameTimeMs : 0.0;
    public double GetLocalGpuLatencyEstimateMs() =>
        HasLocalGpuTiming() ? localRenderTimer.GetLastValidGpuRenderTimeMs() : 0.0;
    public double GetLocalCpuSubmissionTimeMs() => localCpuSubmissionMs;
    public double GetEffectiveLocalLatencyMs() =>
        isLocalNode ? System.Math.Max(frameTimeMs, presentationFrameTimeMs) : frameTimeMs;
    public double GetRTT() => rttMs;         // r：往返时延
    public double GetJitter() => jitterMs;      // j：网络抖动
    public double GetDecodeTime() => decodeTimeMs;  // d：手机解码耗时
    public double GetEncodeTime() => encodeTimeMs;  // c_e：仅发送流时有值
    public double GetBitrate() => bitrateKbps;
    public double GetTransmissionDelay() => transmissionDelayMs;  // t：单帧传输时延
    public double GetAvailableBandwidth() => availableBandwidthMbps; // bw：单位 Mbps
    public bool HasAvailableBandwidth() => availableBandwidthValid;
    public double GetAvailableIncomingBandwidth() => incomingBandwidthValid ? availableIncomingBandwidthMbps : 0.0;
    public double GetAvailableOutgoingBandwidth() => outgoingBandwidthValid ? availableOutgoingBandwidthMbps : 0.0;
    public bool HasAvailableIncomingBandwidth() => incomingBandwidthValid;
    public bool HasAvailableOutgoingBandwidth() => outgoingBandwidthValid;
    public string GetBandwidthSource() => bandwidthSource;
    public void SetExternalAvailableBandwidth(double bandwidthMbps, bool valid)
    {
        if (valid && IsPositiveFinite(bandwidthMbps))
        {
            externalAvailableBandwidthMbps = bandwidthMbps;
            externalBandwidthValid = true;
            lastExternalBandwidthTime = Time.realtimeSinceStartup;

            // Do not wait for the next WebRTC stats coroutine tick after sender BWE arrives.
            if (!incomingBandwidthValid)
            {
                availableBandwidthMbps = bandwidthMbps;
                availableBandwidthValid = true;
                bandwidthSource = "sender-udp-bwe";
                transmissionDelayMs = CalculateTransmissionDelayMs(
                    availableBandwidthMbps, out transmissionDelayStatus);
            }
        }
        else
        {
            externalAvailableBandwidthMbps = 0.0;
            externalBandwidthValid = false;
            lastExternalBandwidthTime = float.NegativeInfinity;
        }
    }

    public double EstimateTransmissionDelayMs(double bandwidthMbps)
    {
        return CalculateTransmissionDelayMs(bandwidthMbps, out _);
    }

    private double CalculateTransmissionDelayMs(double bandwidthMbps, out string status)
    {
        if (!IsPositiveFinite(bandwidthMbps))
        {
            status = "0: no-valid-bandwidth";
            return 0.0;
        }

        float frameSampleMaxAge = Mathf.Max(1.0f, bandwidthHoldSeconds);
        bool hasFreshFrameSize =
            IsPositiveFinite(averageReceivedFrameBytes) &&
            Time.realtimeSinceStartup - lastFrameSizeSampleTime <= frameSampleMaxAge;

        if (hasFreshFrameSize)
        {
            status = "frame-bytes / sender-bwe";
            double frameBits = averageReceivedFrameBytes * 8.0;
            return (frameBits / (bandwidthMbps * 1_000_000.0)) * 1000.0;
        }

        double streamFps = receiveFps > 0.0 ? receiveFps : fps;
        if (bitrateKbps <= 0)
        {
            status = "0: no-received-bitrate";
            return 0.0;
        }

        if (streamFps <= 0)
        {
            status = "0: no-receive-fps";
            return 0.0;
        }

        status = "bitrate / fps fallback";
        double fallbackFrameBits = (bitrateKbps * 1000.0) / streamFps;
        return (fallbackFrameBits / (bandwidthMbps * 1_000_000.0)) * 1000.0;
    }
}
