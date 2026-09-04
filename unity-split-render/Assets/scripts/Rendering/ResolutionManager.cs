using UnityEngine;

public class ResolutionManager : MonoBehaviour
{
    void Start()
    {
        // 强制设置为 1920宽, 1080高, 全屏
        Screen.SetResolution(1920, 1080, true);
    }
}