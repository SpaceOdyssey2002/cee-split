// MultiStreamReceiver.cs
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using Unity.RenderStreaming;

public class MultiStreamReceiver : MonoBehaviour
{
    [System.Serializable]
    public class StreamSlot
    {
        public string connectionId;
        public SingleConnection connection;
        public VideoStreamReceiver videoReceiver;
        public RawImage displayImage;
    }

    [SerializeField] private SignalingManager renderStreaming;
    [SerializeField] private StreamSlot[] streams;

    // Sender 发出 offer 需要时间，这个值要足够大
    // 建议先设 8 秒，稳定后再调小
    [SerializeField] private float waitForSenderOffer = 8f;

    void Start()
    {
        if (!renderStreaming.Running)
            renderStreaming.Run();

        foreach (var slot in streams)
        {
            var s = slot;
            s.videoReceiver.OnUpdateReceiveTexture += tex =>
            {
                s.displayImage.texture = tex;
                Debug.Log($"[Receiver] ✅ 收到画面: {s.connectionId} {tex.width}x{tex.height}");
            };
        }

        StartCoroutine(ConnectAll());
    }

    IEnumerator ConnectAll()
    {
        yield return new WaitUntil(() => renderStreaming.Running);

        // 等 Sender 完成连接并发出 offer
        Debug.Log($"[Receiver] 等待 {waitForSenderOffer} 秒，让 Sender 先建立连接...");
        yield return new WaitForSeconds(waitForSenderOffer);

        foreach (var slot in streams)
        {
            Debug.Log($"[Receiver] CreateConnection: {slot.connectionId}");
            slot.connection.CreateConnection(slot.connectionId);
            yield return new WaitForSeconds(1.5f);
        }
    }

    void OnDestroy()
    {
        foreach (var slot in streams)
            slot.connection?.DeleteConnection(slot.connectionId);
    }
}
