using Unity.MLAgents.Sensors;
using UnityEngine;

public class CameraTextureSensor : ISensor
{
    private Camera camera;
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private string sensorName;
    private int width;
    private int height;
    private ObservationSpec observationSpec;

    public CameraTextureSensor(Camera camera, RenderTexture renderTexture, string name, int width, int height)
    {
        this.camera = camera;
        this.renderTexture = renderTexture;
        this.sensorName = name;
        this.width = width;
        this.height = height;
        this.texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);
        observationSpec = ObservationSpec.Visual(3, height, width);

    }

    public ObservationSpec GetObservationSpec() => observationSpec;
    public string GetName() => sensorName;
    public byte[] GetCompressedObservation() => null;
    public CompressionSpec GetCompressionSpec() => CompressionSpec.Default();
    public void Update() => CaptureFrame();
    public void Reset() { }

    private void CaptureFrame()
    {
        if (camera == null || renderTexture == null) return;

        // Recreate texture2D if size mismatch
        if (texture2D == null || texture2D.width != renderTexture.width || texture2D.height != renderTexture.height)
            texture2D = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);

        camera.targetTexture = renderTexture;
        camera.Render();

        RenderTexture previousActive = RenderTexture.active;
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        texture2D.Apply();
        RenderTexture.active = previousActive;
        camera.targetTexture = null;
    }

    public int Write(ObservationWriter writer)
    {
        CaptureFrame();
        int index = 0;
        for (int c = 0; c < 3; c++)          // channels first
        {
            for (int h = height - 1; h >= 0; h--)
            {
                for (int w = 0; w < width; w++)
                {
                    Color pixel = texture2D.GetPixel(w, h);
                    if (c == 0) writer[index] = pixel.r;
                    else if (c == 1) writer[index] = pixel.g;
                    else writer[index] = pixel.b;
                    index++;
                }
            }
        }
        return index;
    }
}