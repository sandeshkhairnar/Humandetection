package com.example.humandetect;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class YoloV5Model {
    private Interpreter interpreter;
    public static final int INPUT_SIZE = 640;
    private Context context;

    public YoloV5Model(Context context) {
        this.context = context;
        try {
            MappedByteBuffer modelFile = FileUtil.loadMappedFile(context, "yolov5s.tflite");
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);
            interpreter = new Interpreter(modelFile, options);
            Log.d("YoloV5Model", "Model loaded successfully");
        } catch (IOException e) {
            Log.e("YoloV5Model", "Error loading model: ", e);
        }
    }

    public List<BoundingBox> detectObjects(Bitmap bitmap) {
        Log.d("YoloV5Model", "Starting object detection");

        // Resize and preprocess the bitmap
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
        TensorImage inputImage = TensorImage.fromBitmap(resizedBitmap);

        // Normalize the image (0-255 to 0-1)
        float[][][][] input = new float[1][INPUT_SIZE][INPUT_SIZE][3];
        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                int px = resizedBitmap.getPixel(x, y);
                input[0][y][x][0] = ((px >> 16) & 0xFF) / 255.0f;
                input[0][y][x][1] = ((px >> 8) & 0xFF) / 255.0f;
                input[0][y][x][2] = (px & 0xFF) / 255.0f;
            }
        }

        // Output tensor
        float[][][] output = new float[1][25200][85];

        // Run inference
        interpreter.run(input, output);

        // Process results
        List<BoundingBox> boundingBoxes = processResults(output, bitmap.getWidth(), bitmap.getHeight());

        return boundingBoxes;
    }

    private List<BoundingBox> processResults(float[][][] output, int originalWidth, int originalHeight) {
        List<BoundingBox> boundingBoxes = new ArrayList<>();

        for (int i = 0; i < output[0].length; i++) {
            float[] detection = output[0][i];
            float confidence = detection[4];

            Log.d("YoloV5Model", "Detection " + i + ": confidence = " + confidence);

            if (confidence > 0.1) {  // Lowered threshold for debugging
                float x = detection[0];
                float y = detection[1];
                float w = detection[2];
                float h = detection[3];

                float x1 = x - w / 2;
                float y1 = y - h / 2;
                float x2 = x + w / 2;
                float y2 = y + h / 2;

                x1 *= originalWidth;
                y1 *= originalHeight;
                x2 *= originalWidth;
                y2 *= originalHeight;

                float maxClassConfidence = 0;
                int maxClassIndex = -1;
                for (int j = 5; j < detection.length; j++) {
                    if (detection[j] > maxClassConfidence) {
                        maxClassConfidence = detection[j];
                        maxClassIndex = j - 5;
                    }
                }

                Log.d("YoloV5Model", "Max class: " + maxClassIndex + ", confidence: " + maxClassConfidence);

                if (maxClassIndex == 0 && maxClassConfidence > 0.1) {  // Assuming 0 is person class
                    boundingBoxes.add(new BoundingBox(x1, y1, x2, y2, confidence));
                    Log.d("YoloV5Model", "Person detected: " + x1 + ", " + y1 + ", " + x2 + ", " + y2);
                }
            }
        }

        Log.d("YoloV5Model", "Total detections: " + boundingBoxes.size());
        return boundingBoxes;
    }
}