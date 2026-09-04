using UnityEngine;
using Unity.RenderStreaming; // 只需要这一个核心命名空间
using System.Linq;

public class ForceVideoCodec : MonoBehaviour
{
    public VideoStreamSender videoSender;
    [Tooltip("Codec used by the current VP9 quality proxy and prototype experiments.")]
    public string preferredCodec = "VP9";

    void Start()
    {
        if (videoSender == null)
            videoSender = GetComponent<VideoStreamSender>();

        // 1. 获取 Render Streaming 层面真正支持的编码器
        var availableCodecs = VideoStreamSender.GetAvailableCodecs();

        string codecList = "📱 这台手机真实支持的视频编码器有: ";
        foreach (var c in availableCodecs)
        {
            codecList += c.mimeType + " | ";
        }
        Debug.LogWarning(codecList);

        // 2. 优先使用与当前质量代理一致的 VP9。
        var targetCodec = availableCodecs.FirstOrDefault(
            c => c.mimeType.IndexOf(preferredCodec, System.StringComparison.OrdinalIgnoreCase) >= 0);

        // 若设备不支持 VP9，则依次回退到 VP8 和 H264，并明确记录实际编码器。
        if (targetCodec == null)
        {
            Debug.LogWarning($"未找到 {preferredCodec}，尝试回退到 VP8。");
            targetCodec = availableCodecs.FirstOrDefault(c => c.mimeType.Contains("VP8"));
        }
        if (targetCodec == null)
            targetCodec = availableCodecs.FirstOrDefault(c => c.mimeType.Contains("H264"));

        // 3. 强行覆写 Sender 的编码器设置！
        if (targetCodec != null)
        {
            videoSender.SetCodec(targetCodec);
            Debug.LogWarning("发送端编码器设置为: " + targetCodec.mimeType);
        }
        else
        {
            Debug.LogError("未找到 VP9、VP8 或 H264 视频编码器。");
        }
    }
}
