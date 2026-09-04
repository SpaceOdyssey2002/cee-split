using System.Collections.Generic;
using System.Globalization;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

/// <summary>
/// Fixed-threshold counterpart of AutoAdaptiveTriCounter.
///
/// This component intentionally keeps the same visibility test, distance
/// definition, triangle counting, layer assignment, update interval, UI, and
/// UDP payload as AutoAdaptiveTriCounter. The only experimental variable is
/// that the Near/Mid and Mid/Far boundaries remain fixed.
/// </summary>
public class FixedThresholdTriCounter : MonoBehaviour
{
    [Header("Core Settings")]
    public Camera mainCam;

    [Header("Fixed Split Thresholds")]
    [Min(0f)] public float fixedSplitNear = 25f;
    [Min(0f)] public float fixedSplitMid = 65f;

    [Header("Runtime Results")]
    public int trisNear;
    public int trisMid;
    public int trisFar;
    public float currentSplit_Near;
    public float currentSplit_Mid;

    [Header("Network and Node")]
    [Tooltip("当前设备上的监控器：云/边端指向本机发送流 monitor，手机端指向本地渲染 monitor。")]
    public RenderStreamingMonitor localMonitor;
    public int myNodeID = 0;
    public string targetIP = "100.113.34.83";
    public int targetPort = 9998;

    [Header("UI Settings")]
    public bool showUI = true;

    private class RenderData
    {
        public Renderer renderer;
        public int triCount;
        public float currentDist;
        public int lastLayer = -1;
    }

    private readonly List<RenderData> allObjects = new List<RenderData>();
    private Plane[] frustumPlanes;
    private float updateTimer;
    private const float UpdateInterval = 0.2f;
    private int layerNear;
    private int layerMid;
    private int layerFar;
    private string uiNear;
    private string uiMid;
    private string uiFar;
    private string uiTotal;
    private UdpClient udpSender;

    public int TotalTris => trisNear + trisMid + trisFar;

    private void Start()
    {
        if (mainCam == null)
            mainCam = Camera.main;

        // The field name means "monitor local to this process", not specifically
        // the phone's Local Node. Sender nodes use it for encode/frame/BWE stats.
        if (localMonitor == null)
            localMonitor = GetComponent<RenderStreamingMonitor>();

        if ((myNodeID == 0 || myNodeID == 1) && localMonitor == null)
        {
            Debug.LogError(
                $"[FixedTriCounter] Node {myNodeID} has no RenderStreamingMonitor. " +
                "UDP stats will be suppressed to avoid overwriting valid sender BWE.");
        }

        layerNear = LayerMask.NameToLayer("RenderNear");
        layerMid = LayerMask.NameToLayer("RenderMid");
        layerFar = LayerMask.NameToLayer("RenderFar");
        if (layerNear < 0 || layerMid < 0 || layerFar < 0)
        {
            Debug.LogError(
                "[FixedTriCounter] RenderNear, RenderMid and RenderFar layers are required.");
            enabled = false;
            return;
        }

        ApplyFixedThresholds();
        udpSender = new UdpClient { EnableBroadcast = true };
        RefreshSceneObjects();
    }

    private void OnValidate()
    {
        fixedSplitNear = Mathf.Max(0f, fixedSplitNear);
        fixedSplitMid = Mathf.Max(fixedSplitNear, fixedSplitMid);
        ApplyFixedThresholds();
    }

    private void ApplyFixedThresholds()
    {
        currentSplit_Near = Mathf.Max(0f, fixedSplitNear);
        currentSplit_Mid = Mathf.Max(currentSplit_Near, fixedSplitMid);
    }

    public void RefreshSceneObjects()
    {
        allObjects.Clear();
        MeshFilter[] meshes = Object.FindObjectsByType<MeshFilter>();

        foreach (MeshFilter meshFilter in meshes)
        {
            if (meshFilter.sharedMesh == null)
                continue;

            Renderer renderer = meshFilter.GetComponent<Renderer>();
            if (renderer == null || !renderer.enabled)
                continue;

            allObjects.Add(new RenderData
            {
                renderer = renderer,
                triCount = meshFilter.sharedMesh.triangles.Length / 3
            });
        }
    }

    private void Update()
    {
        updateTimer += Time.deltaTime;
        if (updateTimer < UpdateInterval)
            return;

        updateTimer = 0f;
        CalculateMetrics();
    }

    private void CalculateMetrics()
    {
        if (mainCam == null)
            return;

        ApplyFixedThresholds();
        frustumPlanes = GeometryUtility.CalculateFrustumPlanes(mainCam);
        Vector3 camPos = mainCam.transform.position;

        int nearTris = 0;
        int midTris = 0;
        int farTris = 0;

        foreach (RenderData obj in allObjects)
        {
            if (obj.renderer == null ||
                !GeometryUtility.TestPlanesAABB(frustumPlanes, obj.renderer.bounds))
                continue;

            Vector3 closest = obj.renderer.bounds.ClosestPoint(camPos);
            obj.currentDist = Vector3.Distance(camPos, closest);

            int targetLayer = obj.currentDist < currentSplit_Near
                ? layerNear
                : obj.currentDist < currentSplit_Mid ? layerMid : layerFar;

            if (obj.lastLayer != targetLayer)
            {
                obj.renderer.gameObject.layer = targetLayer;
                obj.lastLayer = targetLayer;
            }

            if (targetLayer == layerNear)
                nearTris += obj.triCount;
            else if (targetLayer == layerMid)
                midTris += obj.triCount;
            else
                farTris += obj.triCount;
        }

        trisNear = nearTris;
        trisMid = midTris;
        trisFar = farTris;

        uiNear = $"Near: {nearTris / 1000f:F1}k | < {currentSplit_Near:F1}m";
        uiMid =
            $"Mid:  {midTris / 1000f:F1}k | {currentSplit_Near:F1}-{currentSplit_Mid:F1}m";
        uiFar = $"Far:  {farTris / 1000f:F1}k | >= {currentSplit_Mid:F1}m";
        uiTotal = $"Total: {(nearTris + midTris + farTris) / 1000f:F1}k";

        SendStatsViaUDP(nearTris, midTris, farTris);
    }

    private void SendStatsViaUDP(int nearTris, int midTris, int farTris)
    {
        if (udpSender == null)
            return;

        if ((myNodeID == 0 || myNodeID == 1) && localMonitor == null)
            return;

        try
        {
            double encodeTime = localMonitor != null ? localMonitor.GetEncodeTime() : 0.0;
            double frameTime = localMonitor != null ? localMonitor.GetFrameTimeMs() : 0.0;
            double bandwidth = localMonitor != null
                ? localMonitor.GetAvailableOutgoingBandwidth()
                : 0.0;
            int bandwidthValid =
                localMonitor != null && localMonitor.HasAvailableOutgoingBandwidth() ? 1 : 0;
            string json = "{" +
                "\"n\":" + nearTris + "," +
                "\"m\":" + midTris + "," +
                "\"f\":" + farTris + "," +
                "\"node\":" + myNodeID + "," +
                "\"enc\":" + encodeTime.ToString("F1", CultureInfo.InvariantCulture) + "," +
                "\"rft\":" + frameTime.ToString("F1", CultureInfo.InvariantCulture) + "," +
                "\"bw\":" + bandwidth.ToString("F3", CultureInfo.InvariantCulture) + "," +
                "\"bw_valid\":" + bandwidthValid +
                "}";

            byte[] data = Encoding.UTF8.GetBytes(json);
            udpSender.Send(data, data.Length, targetIP, targetPort);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[FixedTriCounter] UDP send failed: " + e.Message);
        }
    }

    private void OnGUI()
    {
        if (!showUI)
            return;

        GUI.Box(new Rect(10, 10, 250, 120), string.Empty);
        GUI.Label(new Rect(20, 15, 230, 20), "Fixed Threshold Partition");
        GUI.Label(new Rect(20, 35, 230, 20), uiNear);
        GUI.Label(new Rect(20, 55, 230, 20), uiMid);
        GUI.Label(new Rect(20, 75, 230, 20), uiFar);
        GUI.Label(new Rect(20, 95, 230, 20), uiTotal);
    }

#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        Camera cameraToDraw = mainCam != null ? mainCam : Camera.main;
        if (cameraToDraw == null)
            return;

        float near = Mathf.Max(0f, fixedSplitNear);
        float mid = Mathf.Max(near, fixedSplitMid);
        Vector3 camPos = cameraToDraw.transform.position;

        Gizmos.color = new Color(0f, 1f, 0f, 0.5f);
        Gizmos.DrawWireSphere(camPos, near);

        Gizmos.color = new Color(1f, 1f, 0f, 0.5f);
        Gizmos.DrawWireSphere(camPos, mid);
    }
#endif

    private void OnDestroy()
    {
        udpSender?.Close();
    }
}
