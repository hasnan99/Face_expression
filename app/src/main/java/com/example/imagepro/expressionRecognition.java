package com.example.imagepro;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import android.widget.ImageView;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class expressionRecognition {
    private Interpreter interpreter;

    //mengatur ukuran input
    private int input;

    //mengatur ukuran tinggi dan lebar
    private int height=0;
    private int width=0;

    private GpuDelegate gpuDelegate=null;

    private CascadeClassifier cascadeClassifier;

    expressionRecognition(AssetManager assetManager, Context context, String modelpath, int inputsize) throws IOException {
        input=inputsize;
        Interpreter.Options options=new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4);
        interpreter= new Interpreter(loadModelfile(assetManager,modelpath),options);
        Log.d("Facial Expression","Model sudah dimuat ");

        try {
            InputStream inputStream=context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            File cascadedir=context.getDir("cascade",context.MODE_PRIVATE);
            File mcascadefile=new File(cascadedir,"haarcascade_frontalface");
            FileOutputStream outputStream=new FileOutputStream(mcascadefile);
            byte[] buffer=new byte[4096];
            int byteread;
            //-1 berarti tidak membaca data apapun
            while ((byteread= inputStream.read(buffer))!=-1){
                outputStream.write(buffer,0,byteread);
            }
            inputStream.close();
            outputStream.close();
            cascadeClassifier=new CascadeClassifier(mcascadefile.getAbsolutePath());
            Log.d("Facial Expression","Classifier sudah dimuat");

        }catch (IOException e){
            e.printStackTrace();
        }
    }

    public Mat recognizeimage(Mat mat_image){

        Core.flip(mat_image.t(),mat_image,1);

        //mengubah gambar menjadi gray scale
        Mat grayscale=new Mat();
        Imgproc.cvtColor(mat_image,grayscale,Imgproc.COLOR_RGBA2GRAY);

        height=grayscale.height();
        width=grayscale.width();

        int facesize=(int)(height*0.1);

        MatOfRect face=new MatOfRect();
        if (cascadeClassifier!=null){
            cascadeClassifier.detectMultiScale(grayscale,face,1.1,2,2,new Size(facesize,facesize),new Size());
        }

        Rect[] facearray=face.toArray();
        for (int i=0;i<facearray.length;i++){
            Imgproc.rectangle(mat_image,facearray[i].tl(),facearray[i].br(),new Scalar(0,255,0,255),2);
            Rect roi=new Rect((int)facearray[i].tl().x,(int)facearray[i].tl().y,
                    ((int) facearray[i].br().x)-(int) (facearray[i].tl().x),
                    ((int) facearray[i].br().y)-(int) (facearray[i].tl().y));

            Mat crop_rgba=new Mat(mat_image,roi);

            Bitmap bitmap=null;
            bitmap=Bitmap.createBitmap(crop_rgba.cols(),crop_rgba.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(crop_rgba,bitmap);
            Bitmap scaledbitmap=Bitmap.createScaledBitmap(bitmap,48,48,false);
            ByteBuffer byteBuffer=convertbitmap(scaledbitmap);

            float[][]emotion=new float[1][1];
            interpreter.run(byteBuffer,emotion);
            Log.d("Facial Expression","Output : "+ Array.get(Array.get(emotion,0),0));

            float ekspresi_v=(float)Array.get(Array.get(emotion,0),0);
            Log.d("Facial expression","Output: "+ekspresi_v);

            String ekspresi=getemotion(ekspresi_v);

            double ukuran_font = 1.5;
            int Ketebalan = 5;
            Scalar warnafont = new Scalar(255, 255, 255);

            Imgproc.putText(mat_image, ekspresi + " (" + ekspresi_v + ")",
                    new Point((int) facearray[i].tl().x + 10, (int) facearray[i].tl().y + 20),
                    Core.FONT_HERSHEY_SIMPLEX, ukuran_font, warnafont, Ketebalan);
        }

        Core.flip(mat_image.t(),mat_image,0);
        return mat_image;
    }

    public Mat recognizephoto(Mat mat_image){

        //mengubah gambar menjadi gray scale
        Mat grayscale=new Mat();
        Imgproc.cvtColor(mat_image,grayscale,Imgproc.COLOR_RGBA2GRAY);

        height=grayscale.height();
        width=grayscale.width();

        int facesize=(int)(height*0.1);

        MatOfRect face=new MatOfRect();
        if (cascadeClassifier!=null){
            cascadeClassifier.detectMultiScale(grayscale,face,1.1,2,2,new Size(facesize,facesize),new Size());
        }

        Rect[] facearray=face.toArray();
        for (int i=0;i<facearray.length;i++){
            Imgproc.rectangle(mat_image,facearray[i].tl(),facearray[i].br(),new Scalar(0,255,0,255),2);
            Rect roi=new Rect((int)facearray[i].tl().x,(int)facearray[i].tl().y,
                    ((int) facearray[i].br().x)-(int) (facearray[i].tl().x),
                    ((int) facearray[i].br().y)-(int) (facearray[i].tl().y));

            Mat crop_rgba=new Mat(mat_image,roi);

            Bitmap bitmap=null;
            bitmap=Bitmap.createBitmap(crop_rgba.cols(),crop_rgba.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(crop_rgba,bitmap);
            Bitmap scaledbitmap=Bitmap.createScaledBitmap(bitmap,48,48,false);
            ByteBuffer byteBuffer=convertbitmap(scaledbitmap);

            float[][]emotion=new float[1][1];
            interpreter.run(byteBuffer,emotion);
            Log.d("Facial Expression","Output : "+ Array.get(Array.get(emotion,0),0));

            float ekspresi_v=(float)Array.get(Array.get(emotion,0),0);
            Log.d("Facial expression","Output: "+ekspresi_v);

            String ekspresi=getemotion(ekspresi_v);


            Imgproc.putText(mat_image, ekspresi,
                    new Point((int) facearray[i].tl().x + 5, (int) facearray[i].tl().y + 50),
                    2,2,new Scalar(255,0,0,255),2);
        }

        return mat_image;
    }

    private String getemotion(float ekspresi_v) {
        String val="";
        if (ekspresi_v>=0 & ekspresi_v<0.5){
            val="Terkejut";
        }
        else if (ekspresi_v>=0.5 & ekspresi_v<1.5){
            val="Takut";
        }
        else if (ekspresi_v>=1.5 & ekspresi_v<2.5){
            val="marah";
        }
        else if (ekspresi_v>=2.5 & ekspresi_v<3.5) {
            val = "Netral";
        }
        else if (ekspresi_v>=3.5 & ekspresi_v<4.5){
            val="sedih";
        }
        else if (ekspresi_v>=4.5 & ekspresi_v<5.5) {
            val = "Jijik";
        }
        else {
            val="Gembira";
        }
        return val;
    }

    private ByteBuffer convertbitmap(Bitmap scaledbitmap) {
        ByteBuffer byteBuffer;
        int size_image=input;
        byteBuffer=ByteBuffer.allocateDirect(4*1*size_image*size_image*3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intvalues=new int[size_image*size_image];
        scaledbitmap.getPixels(intvalues,0,scaledbitmap.getWidth(),0,0,scaledbitmap.getWidth(),scaledbitmap.getHeight());
        int pixel=0;
        for (int i=0;i<size_image;++i){
            for (int j=0;j<size_image;++j){
                final int val=intvalues[pixel++];
                byteBuffer.putFloat((((val>>16)&0xFF))/255.0f);
                byteBuffer.putFloat((((val>>8)&0xFF))/255.0f);
                byteBuffer.putFloat(((val & 0xFF))/255.0f);
            }
        }
        return byteBuffer;
    }

    //fungsi untuk load model
    private MappedByteBuffer loadModelfile(AssetManager assetManager, String modelpath) throws IOException{
        AssetFileDescriptor assetFileDescriptor=assetManager.openFd(modelpath);
        FileInputStream inputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();

        long startoffset=assetFileDescriptor.getStartOffset();
        long declaredlenght=assetFileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startoffset,declaredlenght);
    }


}
