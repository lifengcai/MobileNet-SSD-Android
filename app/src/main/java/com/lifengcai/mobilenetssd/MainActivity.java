package com.lifengcai.mobilenetssd;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.bumptech.glide.load.engine.DiskCacheStrategy;
import com.bumptech.glide.request.RequestOptions;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int USE_PHOTO = 1001;

    private ImageView show_image;
    private TextView result_text;

    private boolean load_result = false;

    private int[] ddims = {1, 3, 300, 300};

    private List<String> resultLabel = new ArrayList<>();

    private MobileNetSSD mobileNetSSD = new MobileNetSSD();


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try{
            initMobileNetSSD();
        } catch (IOException e){
            e.printStackTrace();
            Log.e(TAG, "onCreate: initMobileNetSSD error");
        }
        request_permissions();
        init_view();
        read_labels();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        String image_path;
        RequestOptions options = new RequestOptions().skipMemoryCache(true)
                .diskCacheStrategy(DiskCacheStrategy.NONE);
        if(resultCode == Activity.RESULT_OK){
            switch(requestCode){
                case USE_PHOTO:
                    if(data == null){
                        Log.w(TAG, "onActivityResult: user photo data is null");
                        return;
                    }
                    Uri image_uri = data.getData();
                    image_path = PhotoUtil.get_path_from_URI(MainActivity.this, image_uri);
                    Log.d(TAG, "onActivityResult: Start predict ...");
                    predict(image_path);
                    break;
                default:
                    break;
            }
        }
    }

    private void initMobileNetSSD() throws IOException{
        byte [] param = null;
        byte [] bin = null;
        {
            // read param
            InputStream assetsInputStream = getAssets().open("MobileNetSSD_deploy.param.bin");
            int available = ((InputStream) assetsInputStream).available();
            param = new byte[available];
            int byteCode = assetsInputStream.read(param);
            assetsInputStream.close();
        }
        {
            // read bin
            InputStream assetsInputStream = getAssets().open("MobileNetSSD_deploy.bin");
            int available = assetsInputStream.available();
            bin = new byte[available];
            int byteCode = assetsInputStream.read(bin);
            assetsInputStream.close();
        }

        load_result = mobileNetSSD.Init(param, bin);
        Log.d(TAG, "initMobileNetSSD: load_result: " + load_result);
    }

    private void request_permissions() {
        List<String> permissionList = new ArrayList<>();
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.ACCESS_FINE_LOCATION);
        }
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.READ_PHONE_STATE) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.READ_PHONE_STATE);
        }
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }
        if (!permissionList.isEmpty()) {
            String [] permissions = permissionList.toArray(new String[permissionList.size()]);
            ActivityCompat.requestPermissions(MainActivity.this, permissions, 1);
        }

    }

    private void init_view(){
        show_image = findViewById(R.id.show_image);
        result_text = findViewById(R.id.result_text);
        result_text.setMovementMethod(ScrollingMovementMethod.getInstance());
        Button use_photo = findViewById(R.id.use_photo);
        use_photo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(!load_result){
                    Toast.makeText(MainActivity.this, "never load model",
                            Toast.LENGTH_SHORT).show();
                    return;
                }
                PhotoUtil.use_photo(MainActivity.this, USE_PHOTO);
            }
        });
    }

    private void read_labels(){
        try{
            AssetManager assetManager = getApplicationContext().getAssets();
            BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open("words.txt")));
            String readLine = null;
            while((readLine = reader.readLine()) != null){
                resultLabel.add(readLine);
            }
        } catch (Exception e){
            e.printStackTrace();
            Log.e(TAG, "read_labels: error");
        }
    }

    private void predict(String image_path) {
        // load image
        Bitmap bmp = PhotoUtil.getScaleBitmap(image_path);
        Bitmap rgba = bmp.copy(Bitmap.Config.ARGB_8888, true);

        // resize image
        Bitmap input_bmp = Bitmap.createScaledBitmap(rgba, ddims[2], ddims[3], false);

        try{
            long start = System.currentTimeMillis();
            float [] result = mobileNetSSD.Detect(input_bmp);
            long end  = System.currentTimeMillis();
            Log.d(TAG, "predict: result:  " + Arrays.toString(result));
            long time = end - start;
            Log.d(TAG, "predict: result length:  " + String.valueOf(result.length));

            String show_text = "result：" + Arrays.toString(result) + "\nname：" +
                    resultLabel.get((int) result[0]) + "\nprobability：" + result[1] + "\ntime：" + time + "ms" ;
            result_text.setText(show_text);

            Canvas canvas = new Canvas(rgba);

            Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(5);

            float finalresult[][] = Onedim2Twodim(result);
            Log.d(TAG, "predict: finalresult: " + finalresult);

            int i = 0;
            int num = result.length / 6;

            for( ; i < num; ++i){

                paint.setColor(Color.RED);
                paint.setStyle(Paint.Style.STROKE);//不填充
                paint.setStrokeWidth(5); //线的宽度
                canvas.drawRect(finalresult[i][2] * rgba.getWidth(), finalresult[i][3] * rgba.getHeight(),
                        finalresult[i][4] * rgba.getWidth(), finalresult[i][5] * rgba.getHeight(), paint);

                paint.setColor(Color.YELLOW);
                paint.setStyle(Paint.Style.FILL);
                paint.setStrokeWidth(1); //线的宽度
                canvas.drawText(resultLabel.get((int) finalresult[i][0]) + "\n" + finalresult[i][1],
                        finalresult[i][2]*rgba.getWidth(),finalresult[i][3]*rgba.getHeight(),paint);
            }

            show_image.setImageBitmap(rgba);
        } catch (Exception e){
            e.printStackTrace();
        }
    }

    public static float[][] Onedim2Twodim(float[] input){
        int n = input.length;
        int num = input.length / 6;
        float[][] output = new float[num][6];
        int k = 0;
        for(int i = 0; i < num; ++i){
            int j = 0;
            while(j < 6){
                output[i][j] = input[k];
                ++k;
                ++j;
            }
        }
        return output;
    }
}
