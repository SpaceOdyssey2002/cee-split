using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement; // 🌟 新增：用于获取场景名

[RequireComponent(typeof(Camera))]
public class DataCollectionPipeline : MonoBehaviour
{
    [Header("Look-Around Settings")]
    public int sampleCount = 100;
    public float maxPitch = 30f;

    [Header("Save Settings")]
    public string baseOutputDir = "DatasetOutputs";
    private string currentRunDir; // 🌟 每次运行的实际保存目录

    private readonly float[] resolutionScales = { 0.33f, 0.67f, 1.0f, 1.33f, 2.0f };
    private readonly Vector2Int[] resolutionProfiles =
    {
        new Vector2Int(640, 360),
        new Vector2Int(1280, 720),
        new Vector2Int(1920, 1080),
        new Vector2Int(2560, 1440),
        new Vector2Int(3840, 2160)
    };
    private readonly int[] qpLevels = { 20, 24, 28, 34, 40 };

    private Vector3 fixedPosition;
    private Camera cam;

    void Start()
    {
        cam = GetComponent<Camera>();
        fixedPosition = transform.position;

        // 🌟 核心修改：生成唯一的文件夹名 (场景名 + 时间戳)
        string sceneName = SceneManager.GetActiveScene().name;
        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
        currentRunDir = Path.Combine(baseOutputDir, $"{sceneName}_{timestamp}");

        Directory.CreateDirectory(currentRunDir);
        Debug.Log($"🚀 本次采集将保存至: {currentRunDir}");

        StartCoroutine(CollectDataRoutine());
    }

    IEnumerator CollectDataRoutine()
    {
        // 🌟 CSV 现在保存在子文件夹内
        string csvPath = Path.Combine(currentRunDir, "metadata.csv");

        using (StreamWriter sw = new StreamWriter(csvPath))
        {
            sw.WriteLine("frame_id,pitch,yaw,res_scale,render_width,render_height,qp,gt_image_path,rendered_image_path");

            for (int i = 0; i < sampleCount; i++)
            {
                SetRandomRotation();
                yield return null;

                // 🌟 所有路径都指向 currentRunDir
                string gtPath = Path.Combine(currentRunDir, $"gt_{i}.png");
                CaptureFrameAtResolution(gtPath, resolutionProfiles[resolutionProfiles.Length - 1]);

                for (int profileIndex = 0; profileIndex < resolutionProfiles.Length; profileIndex++)
                {
                    float res = resolutionScales[profileIndex];
                    Vector2Int resolution = resolutionProfiles[profileIndex];
                    string renderPath = Path.Combine(currentRunDir, $"render_{i}_r{res:F2}.png");
                    CaptureFrameAtResolution(renderPath, resolution);

                    float currentPitch = transform.rotation.eulerAngles.x;
                    float currentYaw = transform.rotation.eulerAngles.y;
                    if (currentPitch > 180f) currentPitch -= 360f;

                    foreach (int qp in qpLevels)
                    {
                        // 🌟 写入 CSV 的路径也要包含子文件夹名，方便后面 Python 读取
                        sw.WriteLine(
                            $"{i},{currentPitch:F2},{currentYaw:F2},{res}," +
                            $"{resolution.x},{resolution.y},{qp},{gtPath},{renderPath}");
                    }
                }
                Debug.Log($"Progress: {i + 1} / {sampleCount} (Scene: {SceneManager.GetActiveScene().name})");
            }
        }
        Debug.Log($"✅ 采集完成！存放在: {currentRunDir}");
    }

    private void CaptureFrameAtResolution(string path, Vector2Int resolution)
    {
        int renderWidth = resolution.x;
        int renderHeight = resolution.y;

        RenderTexture rtInternal = RenderTexture.GetTemporary(renderWidth, renderHeight, 24, RenderTextureFormat.ARGB32);
        rtInternal.filterMode = FilterMode.Bilinear;
        RenderTexture previousActive = RenderTexture.active;
        RenderTexture previousTarget = cam.targetTexture;
        cam.targetTexture = rtInternal;
        cam.Render();

        RenderTexture.active = rtInternal;
        Texture2D tex = new Texture2D(renderWidth, renderHeight, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, renderWidth, renderHeight), 0, 0);
        tex.Apply();

        cam.targetTexture = previousTarget;
        RenderTexture.active = previousActive;
        RenderTexture.ReleaseTemporary(rtInternal);

        File.WriteAllBytes(path, tex.EncodeToPNG());
        Destroy(tex);
    }

    private void SetRandomRotation()
    {
        transform.position = fixedPosition;
        float randomYaw = Random.Range(0f, 360f);
        float randomPitch = Random.Range(-maxPitch, maxPitch);
        transform.rotation = Quaternion.Euler(randomPitch, randomYaw, 0f);
    }
}
