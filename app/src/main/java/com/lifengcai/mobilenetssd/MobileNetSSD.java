package com.lifengcai.mobilenetssd;

import android.graphics.Bitmap;

public class MobileNetSSD {

    public native boolean Init(byte[] param, byte[] bin);
    public native float[] Detect(Bitmap bitmap);

    static{
        System.loadLibrary("MobileNetSSD");
    }
}
