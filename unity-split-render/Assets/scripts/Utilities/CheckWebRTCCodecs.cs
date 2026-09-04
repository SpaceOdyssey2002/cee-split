using UnityEngine;
using Unity.WebRTC;
using System.Collections;
using System.Text;

public class CheckWebRTCCodecs : MonoBehaviour
{
    private string codecInfo = "正在获取编解码器...";

    void Start()
    {
        // 确保在 WebRTC 初始化之后再获取
        StartCoroutine(GetCodecsRoutine());
    }

    IEnumerator GetCodecsRoutine()
    {
        // 等待 Render Streaming 初始化 WebRTC
        yield return new WaitForSeconds(2.0f);

        StringBuilder sb = new StringBuilder();
        sb.AppendLine("=== 📱 手机支持的视频解码器 (Receiver) ===");

        // 获取当前设备 WebRTC 支持的接收（解码）能力
        var receiverCaps = RTCRtpReceiver.GetCapabilities(TrackKind.Video);

        if (receiverCaps.codecs != null)
        {
            foreach (var codec in receiverCaps.codecs)
            {
                // 只看视频相关的
                if (codec.mimeType.Contains("video"))
                {
                    sb.AppendLine($"[Codec] {codec.mimeType} | 频段: {codec.clockRate}");
                    if (!string.IsNullOrEmpty(codec.sdpFmtpLine))
                    {
                        // 这一行极其重要，它决定了 H264 的 Profile (Baseline/Main/High)
                        sb.AppendLine($"   -> 参数: {codec.sdpFmtpLine}");
                    }
                }
            }
        }
        else
        {
            sb.AppendLine("未检测到解码器能力，WebRTC 可能未正确初始化。");
        }

        codecInfo = sb.ToString();
        Debug.Log(codecInfo);
    }

    // 画在屏幕左上角，方便你在手机上直接看
    void OnGUI()
    {
        GUIStyle style = new GUIStyle();
        style.fontSize = 20;
        style.normal.textColor = Color.yellow;
        GUI.Label(new Rect(10, 10, Screen.width, Screen.height), codecInfo, style);
    }
}