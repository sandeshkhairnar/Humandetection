package com.example.humandetect;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity implements LifecycleOwner {

    private static final String MODEL_PATH = "yolov5s.tflite";
    private static final int INPUT_SIZE = 640;
    private static final int NUM_CLASSES = 80; // Adjust based on your model
    private static final int OUTPUT_WIDTH = 8400; // Adjust based on your model's output

    private Interpreter interpreter;
    private TextView textViewResults;
    private PreviewView previewView;
    private Button buttonSwitchCamera;
    private ImageView resultImageView;
    private boolean isFrontCamera = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textViewResults = findViewById(R.id.textViewResults);
        previewView = findViewById(R.id.previewView);
        buttonSwitchCamera = findViewById(R.id.buttonSwitchCamera);
        resultImageView = findViewById(R.id.resultImageView);

        try {
            interpreter = new Interpreter(loadModelFile());
            Log.d("MainActivity", "TensorFlow Lite model loaded successfully");
        } catch (IOException e) {
            Log.e("MainActivity", "Error loading TensorFlow Lite model", e);
            textViewResults.setText("Error: Could not load TensorFlow Lite model");
            return;
        }

        buttonSwitchCamera.setOnClickListener(v -> switchCamera());
        startCamera();
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(INPUT_SIZE, INPUT_SIZE))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
                imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), this::processImageProxy);

                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(isFrontCamera ? CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK)
                        .build();

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);

            } catch (ExecutionException | InterruptedException e) {
                Log.e("MainActivity", "Error starting camera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void switchCamera() {
        isFrontCamera = !isFrontCamera;
        startCamera();
    }

    private void processImageProxy(@NonNull ImageProxy imageProxy) {
        try {
            Bitmap bitmap = toBitmap(imageProxy);
            if (bitmap != null) {
                // Resize the bitmap to the model's expected input size
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
                List<Detection> detections = detectHuman(resizedBitmap);
                drawBoundingBoxes(resizedBitmap, detections);
            }
        } catch (Exception e) {
            Log.e("MainActivity", "Error processing image", e);
        } finally {
            imageProxy.close();
        }
    }

    private List<Detection> detectHuman(Bitmap bitmap) {
        if (interpreter == null) {
            Log.e("MainActivity", "Interpreter is null. Model may not have been loaded correctly.");
            return new ArrayList<>();
        }

        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * Float.BYTES);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int i = 0; i < intValues.length; i++) {
            final int val = intValues[i];
            byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
            byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
            byteBuffer.putFloat((val & 0xFF) / 255.0f);
        }

        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, INPUT_SIZE, INPUT_SIZE, 3}, DataType.FLOAT32);
        inputFeature0.loadBuffer(byteBuffer);

        TensorBuffer outputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 84, OUTPUT_WIDTH}, DataType.FLOAT32);
        interpreter.run(inputFeature0.getBuffer(), outputFeature0.getBuffer());

        float[][] outputs = new float[84][OUTPUT_WIDTH];
        for (int i = 0; i < 84; i++) {
            for (int j = 0; j < OUTPUT_WIDTH; j++) {
                outputs[i][j] = outputFeature0.getFloatValue(i * OUTPUT_WIDTH + j);
            }
        }

        return processDetections(outputs);
    }

    private List<Detection> processDetections(float[][] outputs) {
        List<Detection> detections = new ArrayList<>();
        float confidenceThreshold = 0.5f;
        float iouThreshold = 0.5f;

        for (int i = 0; i < OUTPUT_WIDTH; i++) {
            float confidence = outputs[4][i];
            if (confidence > confidenceThreshold) {
                float x = outputs[0][i];
                float y = outputs[1][i];
                float w = outputs[2][i];
                float h = outputs[3][i];

                float xmin = x - w / 2;
                float ymin = y - h / 2;
                float xmax = x + w / 2;
                float ymax = y + h / 2;

                int classId = 0;
                float maxClassScore = 0;
                for (int j = 5; j < 85; j++) {
                    if (outputs[j][i] > maxClassScore) {
                        maxClassScore = outputs[j][i];
                        classId = j - 5;
                    }
                }

                RectF boundingBox = new RectF(xmin, ymin, xmax, ymax);
                detections.add(new Detection(boundingBox, classId, confidence));
            }
        }

        // Implement Non-Maximum Suppression here if needed

        return detections;
    }

    private Bitmap toBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private void drawBoundingBoxes(Bitmap bitmap, List<Detection> detections) {
        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5f);

        for (Detection detection : detections) {
            paint.setColor(getColorForClass(detection.classId));
            canvas.drawRect(detection.boundingBox, paint);

            String label = getClassLabel(detection.classId) + " " + String.format("%.2f", detection.confidence);
            paint.setTextSize(30f);
            canvas.drawText(label, detection.boundingBox.left, detection.boundingBox.top - 10, paint);
        }

        runOnUiThread(() -> {
            resultImageView.setImageBitmap(mutableBitmap);
            textViewResults.setText("Detected: " + detections.size() + " objects");
        });
    }

    private int getColorForClass(int classId) {
        // Implement a color selection scheme for different classes
        return Color.RED; // Default color
    }

    private String getClassLabel(int classId) {
        // Implement a mapping from class ID to human-readable label
        return "Class " + classId;
    }

    private static class Detection {
        RectF boundingBox;
        int classId;
        float confidence;

        Detection(RectF boundingBox, int classId, float confidence) {
            this.boundingBox = boundingBox;
            this.classId = classId;
            this.confidence = confidence;
        }
    }
}