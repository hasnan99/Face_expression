package com.example.imagepro;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;

public class gallery_activity extends AppCompatActivity {

    private Button pilih_gambar;
    private ImageView gambar_v;

    int select_picture=200;

    private expressionRecognition expressionRecognition;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery);

        pilih_gambar=findViewById(R.id.pilih_gambar);
        gambar_v=findViewById(R.id.gambar_v);

        try {
            int input = 48;
            expressionRecognition = new expressionRecognition(getAssets(),gallery_activity.this,
                    "newmodel.tflite",input);
        }catch (IOException e){
            e.printStackTrace();
        }

        pilih_gambar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                select_image();
            }
        });

    }

    private void select_image() {
        Intent intent= new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent,"Pilih Gambar"),select_picture);
    }

    @Override
    public void onActivityResult(int requestcode,int resultcode,Intent data){
        super.onActivityResult(requestcode,resultcode,data);
        if (resultcode==RESULT_OK){
            if (requestcode==select_picture){
                Uri selectimageuri=data.getData();
                if (selectimageuri!=null){
                    Log.d("Gallery activity","Output Uri:"+selectimageuri);

                    Bitmap bitmap=null;
                    try {
                        bitmap= MediaStore.Images.Media.getBitmap(this.getContentResolver(),selectimageuri);
                    }
                    catch (IOException e){
                        e.printStackTrace();
                    }
                    Mat selected_image=new Mat(bitmap.getHeight(),bitmap.getWidth(), CvType.CV_8UC4);
                    Utils.bitmapToMat(bitmap,selected_image);

                    selected_image=expressionRecognition.recognizephoto(selected_image);

                    Bitmap bitmap1=null;
                    bitmap1=Bitmap.createBitmap(selected_image.cols(),selected_image.rows(),Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(selected_image,bitmap1);
                    gambar_v.setImageBitmap(bitmap1);


                }
            }
        }
    }
}