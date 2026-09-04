Shader "Custom/DepthCompositePacked"
{
    Properties
    {
        _NearTex ("Near Packed", 2D) = "black" {}
        _MidTex  ("Mid Packed",  2D) = "black" {}
        _FarTex  ("Far Packed",  2D) = "black" {}
        _HasNear ("Has Near", Float) = 0
        _HasMid  ("Has Mid",  Float) = 0
        _HasFar  ("Has Far",  Float) = 0
        _BackgroundDepthThreshold ("Background Depth Threshold", Range(0.9, 1.0)) = 0.995
    }
    SubShader
    {
        Pass
        {
            ZTest Always Cull Off ZWrite Off
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _NearTex, _MidTex, _FarTex;
            float _HasNear, _HasMid, _HasFar;
            float2 _CameraClip;
            float _BackgroundDepthThreshold;

            struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; };
            struct v2f { float4 vertex : SV_POSITION; float2 uv : TEXCOORD0; };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float PackedToEyeDepth(float encodedDepth)
            {
                float nearPlane = max(_CameraClip.x, 1e-4);
                float farPlane = max(_CameraClip.y, nearPlane + 1e-3);
                return nearPlane * exp2(
                    saturate(encodedDepth) * log2(farPlane / nearPlane));
            }

            void SamplePackedLayer(
                sampler2D packedTex,
                float2 uv,
                out fixed3 rgb,
                out float eyeDepth,
                out float isBackground)
            {
                float2 uvRgb = float2(uv.x, 0.5 + uv.y * 0.5);
                float2 uvDepth = float2(uv.x, uv.y * 0.5);
                rgb = tex2D(packedTex, uvRgb).rgb;
                float encodedDepth = tex2D(packedTex, uvDepth).r;
                eyeDepth = PackedToEyeDepth(encodedDepth);
                isBackground =
                    encodedDepth >= _BackgroundDepthThreshold ? 1.0 : 0.0;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                fixed3 rgbN, rgbM, rgbF;
                float depN, depM, depF;
                float bgN, bgM, bgF;

                SamplePackedLayer(_NearTex, i.uv, rgbN, depN, bgN);
                SamplePackedLayer(_MidTex, i.uv, rgbM, depM, bgM);
                SamplePackedLayer(_FarTex, i.uv, rgbF, depF, bgF);

                const float INF = 1e9;
                bool skyN = (_HasNear < 0.5 || bgN > 0.5);
                bool skyM = (_HasMid < 0.5 || bgM > 0.5);
                bool skyF = (_HasFar < 0.5 || bgF > 0.5);

                float wN = skyN ? INF : depN;
                float wM = skyM ? INF : depM;
                float wF = skyF ? INF : depF;

                if (skyN && skyM && skyF)
                    return fixed4(rgbF, 1);

                float bestDepth = INF;
                fixed3 finalColor = rgbF;

                if (!skyF)
                {
                    bestDepth = wF;
                    finalColor = rgbF;
                }

                if (!skyM && wM < bestDepth)
                {
                    bestDepth = wM;
                    finalColor = rgbM;
                }

                if (!skyN && wN < bestDepth)
                    finalColor = rgbN;

                return fixed4(finalColor, 1);
            }
            ENDCG
        }
    }
}
