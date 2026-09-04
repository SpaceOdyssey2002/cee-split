using UnityEngine;

public class PingPongMove : MonoBehaviour
{
    public float speed = 3.0f;    // 移动速度
    public float distance = 5.0f; // 移动距离
    private Vector3 startPos;

    void Start()
    {
        startPos = transform.position;
    }

    void Update()
    {
        // Mathf.PingPong 会随时间在 0 到 distance 之间循环
        float move = Mathf.PingPong(Time.time * speed, distance);

        // 改变 X 轴位置（你也可以改 Y 或 Z）
        transform.position = startPos + new Vector3(move, 0, 0);
    }
}