using UnityEngine;
using Unity.RenderStreaming;
using System.Collections;

public class FixedIdSender : MonoBehaviour
{
    [SerializeField] private SignalingManager renderStreaming;
    [SerializeField] private SingleConnection connection;
    [SerializeField] private string connectionId = "stream-near"; [Header("本地节点开关")]
    public bool isLocalNode = false; // 👈 手机上勾选这个！

    void Start()
    {
        if (isLocalNode) return; // 👈 如果是手机本地节点，直接退出，不连 WebRTC

        if (!renderStreaming.Running)
            renderStreaming.Run();

        StartCoroutine(Connect());
    }

    IEnumerator Connect()
    {
        yield return new WaitUntil(() => renderStreaming.Running);

        connection.CreateConnection(connectionId);
        Debug.Log($"[Sender] ✅ polite:false 应该是这个 → {connectionId}");
    }

    void OnDestroy()
    {
        if (!isLocalNode)
        {
            connection.DeleteConnection(connectionId);
        }
    }
}