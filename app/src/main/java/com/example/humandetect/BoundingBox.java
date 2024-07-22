package com.example.humandetect;

public class BoundingBox {
    public float x1, y1, x2, y2, confidence;

    public BoundingBox(float x1, float y1, float x2, float y2, float confidence) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.confidence = confidence;
    }
}