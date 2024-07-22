package com.example.humandetect;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
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
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void detectObjects(Bitmap bitmap) {
        TensorImage inputImage = TensorImage.fromBitmap(bitmap);

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .add(getPreprocessNormalizeOp())
                .build();

        inputImage = imageProcessor.process(inputImage);

        float[][][][] input = new float[1][INPUT_SIZE][INPUT_SIZE][3];
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        inputImage.getBitmap().getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                input[0][i][j][0] = ((pixelValue >> 16) & 0xFF) / 255.0f;
                input[0][i][j][1] = ((pixelValue >> 8) & 0xFF) / 255.0f;
                input[0][i][j][2] = (pixelValue & 0xFF) / 255.0f;
            }
        }

        float[][][] output = new float[1][25200][85];
        interpreter.run(input, output);

        List<BoundingBox> boundingBoxes = processResults(output);
        if (context instanceof MainActivity) {
            ((MainActivity) context).runOnUiThread(() ->
                    ((MainActivity) context).updateBoundingBoxes(boundingBoxes)
            );
        }
    }

    private TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(0f, 255f);
    }

    private List<BoundingBox> processResults(float[][][] output) {
        List<BoundingBox> boundingBoxes = new ArrayList<>();

        for (float[] detection : output[0]) {
            float confidence = detection[4];
            Log.d("YoloV5Model", "Detection confidence: " + confidence);

            // Log full detection result for inspection
            Log.d("YoloV5Model", "Detection result: " + Arrays.toString(detection));

            if (confidence > 0.3) { // Adjusted threshold
                float x = detection[0];
                float y = detection[1];
                float w = detection[2];
                float h = detection[3];

                float x1 = x - w / 2f;
                float y1 = y - h / 2f;
                float x2 = x + w / 2f;
                float y2 = y + h / 2f;

                int classIndex = (int) detection[5];
                Log.d("YoloV5Model", "Class index: " + classIndex);
                if (classIndex == 0) { // Assuming human class index is 0
                    boundingBoxes.add(new BoundingBox(x1, y1, x2, y2));
                    Log.d("YoloV5Model", "Human detected: " + x1 + ", " + y1 + ", " + x2 + ", " + y2);
                }
            }
        }

        Log.d("YoloV5Model", "Total detections: " + boundingBoxes.size());
        return boundingBoxes;
    }

}