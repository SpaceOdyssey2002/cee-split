using UnityEngine;
using System.Collections.Generic;

public class FixedDistanceTriCounter : MonoBehaviour
{
    [Header("核心设置")]
    public Camera mainCam;

    [Header("固定分层距离")]
    public float splitNear = 10f;   // Near/Mid 分界距离（米）
    public float splitMid = 30f;   // Mid/Far  分界距离（米）
    //public float splitNear = 20f;   // Near/Mid 分界距离（米）
    //public float splitMid = 60f;   // Mid/Far  分界距离（米）

    [Header("实时结果 (仅供监视)")]
    public int trisNear, trisMid, trisFar;

    private class RenderData
    {
        public Renderer renderer;
        public int triCount;
        public float currentDist;
        public int lastLayer = -1;
    }

    private List<RenderData> allObjects = new List<RenderData>();
    private Plane[] frustumPlanes;
    private float updateTimer;
    private const float updateInterval = 0.2f;
    private int layerNear, layerMid, layerFar;
    public int TotalTris => trisNear + trisMid + trisFar;

    private string uiNear, uiMid, uiFar, uiTotal;

    void Start()
    {
        if (mainCam == null) mainCam = Camera.main;
        layerNear = LayerMask.NameToLayer("RenderNear");
        layerMid = LayerMask.NameToLayer("RenderMid");
        layerFar = LayerMask.NameToLayer("RenderFar");
        RefreshSceneObjects();
    }

    public void RefreshSceneObjects()
    {
        allObjects.Clear();
        MeshFilter[] meshes = Object.FindObjectsByType<MeshFilter>(FindObjectsSortMode.None);
        foreach (var mf in meshes)
        {
            if (mf.sharedMesh == null) continue;
            Renderer r = mf.GetComponent<Renderer>();
            if (r == null || !r.enabled) continue;
            allObjects.Add(new RenderData
            {
                renderer = r,
                triCount = mf.sharedMesh.triangles.Length / 3
            });
        }
    }

    void Update()
    {
        updateTimer += Time.deltaTime;
        if (updateTimer < updateInterval) return;
        updateTimer = 0f;
        CalculateMetrics();
    }

    void CalculateMetrics()
    {
        if (mainCam == null) return;
        frustumPlanes = GeometryUtility.CalculateFrustumPlanes(mainCam);
        Vector3 camPos = mainCam.transform.position;

        // splitNear < splitMid を保証
        float near = splitNear;
        float mid = Mathf.Max(splitMid, splitNear);

        int tn = 0, tm = 0, tf = 0;

        foreach (var obj in allObjects)
        {
            if (obj.renderer == null) continue;
            if (!GeometryUtility.TestPlanesAABB(frustumPlanes, obj.renderer.bounds)) continue;

            Vector3 closest = obj.renderer.bounds.ClosestPoint(camPos);
            obj.currentDist = Vector3.Distance(camPos, closest);

            int target = (obj.currentDist < near) ? layerNear :
                         (obj.currentDist < mid) ? layerMid : layerFar;

            if (obj.lastLayer != target)
            {
                obj.renderer.gameObject.layer = target;
                obj.lastLayer = target;
            }

            if (target == layerNear) tn += obj.triCount;
            else if (target == layerMid) tm += obj.triCount;
            else tf += obj.triCount;
        }

        trisNear = tn; trisMid = tm; trisFar = tf;

        uiNear = $"Near: {tn / 1000f:F1}k  (<{near:F0}m)";
        uiMid = $"Mid:  {tm / 1000f:F1}k  ({near:F0}~{mid:F0}m)";
        uiFar = $"Far:  {tf / 1000f:F1}k  (>{mid:F0}m)";
        uiTotal = $"Total: {(tn + tm + tf) / 1000f:F1}k";
    }

    void OnGUI()
    {
        GUI.Box(new Rect(10, 10, 240, 100), "");
        GUI.Label(new Rect(20, 15, 220, 20), uiNear);
        GUI.Label(new Rect(20, 35, 220, 20), uiMid);
        GUI.Label(new Rect(20, 55, 220, 20), uiFar);
        GUI.Label(new Rect(20, 75, 220, 20), uiTotal);
    }

#if UNITY_EDITOR
    // Scene视图中可视化分层边界
    void OnDrawGizmosSelected()
    {
        if (mainCam == null) return;
        Vector3 pos = mainCam.transform.position;

        Gizmos.color = new Color(0f, 1f, 0f, 0.3f);
        Gizmos.DrawWireSphere(pos, splitNear);

        Gizmos.color = new Color(1f, 1f, 0f, 0.2f);
        Gizmos.DrawWireSphere(pos, Mathf.Max(splitMid, splitNear));
    }
#endif
}