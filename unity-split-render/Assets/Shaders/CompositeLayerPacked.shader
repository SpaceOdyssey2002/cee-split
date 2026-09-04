Shader "Custom/CompositeLayerPacked"
{
    Properties
    {
        _MainTex ("Blit Source", 2D) = "black" {}
        _NearTex ("Near Packed", 2D) = "black" {}
        _MidTex  ("Mid Packed", 2D) = "black" {}
        _FarTex  ("Far Packed", 2D) = "black" {}
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        ZTest Always Cull Off ZWrite Off

        Pass
        {
            Name "ScaleAndCompositePackedLayers"

            CGPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            sampler2D _NearTex;
            sampler2D _MidTex;
            sampler2D _FarTex;
            float _NearActive;
            float _MidActive;
            float _FarActive;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata input)
            {
                v2f output;
                output.vertex = UnityObjectToClipPos(input.vertex);
                output.uv = input.uv;
                return output;
            }

            void SampleLayer(
                sampler2D layerTexture,
                float2 viewportUV,
                out fixed4 color,
                out float depth)
            {
                // PackRgbDepth and DepthCompositePacked use the same fixed
                // convention: RGB occupies the upper half and logarithmic depth
                // occupies the lower half. This conventional Graphics.Blit
                // vertex path preserves that layout on D3D, Vulkan and GLES.
                float2 colorUV = float2(viewportUV.x, 0.5 + 0.5 * viewportUV.y);
                float2 depthUV = float2(viewportUV.x, 0.5 * viewportUV.y);
                color = tex2D(layerTexture, colorUV);
                depth = tex2D(layerTexture, depthUV).r;
            }

            fixed4 frag(v2f input) : SV_Target
            {
                bool outputColor = input.uv.y >= 0.5;
                float2 viewportUV = float2(
                    input.uv.x,
                    outputColor ? (input.uv.y - 0.5) * 2.0 : input.uv.y * 2.0);

                fixed4 nearColor = fixed4(0, 0, 0, 0);
                fixed4 midColor = fixed4(0, 0, 0, 0);
                fixed4 farColor = fixed4(0, 0, 0, 0);
                float nearDepth = 1.0;
                float midDepth = 1.0;
                float farDepth = 1.0;

                if (_NearActive > 0.5)
                    SampleLayer(_NearTex, viewportUV, nearColor, nearDepth);
                if (_MidActive > 0.5)
                    SampleLayer(_MidTex, viewportUV, midColor, midDepth);
                if (_FarActive > 0.5)
                    SampleLayer(_FarTex, viewportUV, farColor, farDepth);

                fixed4 selectedColor = _FarActive > 0.5 ? farColor
                    : (_MidActive > 0.5 ? midColor : nearColor);
                float selectedDepth = _FarActive > 0.5 ? farDepth
                    : (_MidActive > 0.5 ? midDepth : nearDepth);

                if (_MidActive > 0.5 && midDepth < selectedDepth)
                {
                    selectedDepth = midDepth;
                    selectedColor = midColor;
                }
                if (_NearActive > 0.5 && nearDepth < selectedDepth)
                {
                    selectedDepth = nearDepth;
                    selectedColor = nearColor;
                }

                return outputColor
                    ? selectedColor
                    : fixed4(selectedDepth, selectedDepth, selectedDepth, 1.0);
            }
            ENDCG
        }
    }
}
