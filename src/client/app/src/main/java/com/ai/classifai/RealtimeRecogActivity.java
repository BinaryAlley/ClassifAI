package com.ai.classifai;

// #region ================================================================== IMPORTS ====================================================================================
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatImageView;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.google.android.material.chip.Chip;
import com.google.common.util.concurrent.ListenableFuture;
import com.ai.classifai.helpers.Persistence;
import com.ai.classifai.models.RecognitionResultModel;
import com.ai.classifai.tflite.Classifier;
import org.tensorflow.lite.examples.classification.R;
import com.ai.classifai.customview.RecognitionResultView;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import static android.Manifest.permission.CAMERA;
// #endregion

/**
 * Activity for real-time recognition using the camera
 * 
 * Creation Date: 09th of January, 2021
 */
public class RealtimeRecogActivity extends AppCompatActivity {
// #region =============================================================== FIELD MEMBERS =================================================================================
    private static final int REQUEST_CODE_CAMERA = 100;
    ExecutorService cameraExecutor;
    RecognitionResultModel currentResult;
    Bitmap currentBitmap;
    Classifier classifier;
    PreviewView camera;
    boolean isCameraOpened = false;
    Chip btnResume, btnSave;
    AppCompatImageView imgPicked;
    RecognitionResultView resultView;
// #endregion    

// #region ================================================================== METHODS ====================================================================================
    /**
     * Sets up the activity, initializes UI components and classifier
     *
     * @param savedInstanceState If the activity is being re-initialized after being previously shut down, this Bundle contains the data it most recently supplied
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // set the user interface layout for this Activity
        setContentView(R.layout.activity_real_time_recog);
        // executor for running camera operations in the background
        cameraExecutor = Executors.newSingleThreadExecutor();
        // initialize UI components for displaying recognition results and controls
        resultView = findViewById(R.id.rt_result_view);
        btnResume = findViewById(R.id.btn_resume);
        btnSave = findViewById(R.id.btn_save);
        camera = findViewById(R.id.camera);
        imgPicked = findViewById(R.id.img_picked);
        // set up click listener for the save button
        btnSave.setOnClickListener(v -> {
            // check and request necessary permissions before saving
            if(Persistence.checkPermissions(this)){
                // save the current prediction and notify user
                boolean saved = Persistence.self().savePrediction(currentBitmap, currentResult);
                if (saved){
                    Toast.makeText(this,"Prediction Saved.",Toast.LENGTH_SHORT).show();
                    // reopen camera for further recognition
                    openCamera();
                }
            }
            else
                Persistence.requestPermissions(this);
        });
        btnResume.setOnClickListener(v -> {
            openCamera(); // open camera to resume recognition
        });
        // initialize classifier used for image recognition
        try {
            classifier = Classifier.create(this, Classifier.Device.CPU,-1);
        } catch (IOException e) {
            e.printStackTrace();
        }
        // check for camera permissions and set up camera if granted
        if(allPermissionsGranted()){
            connectCamera();
            openCamera();
        }
        else 
            // request camera permissions if not already granted
            ActivityCompat.requestPermissions(this,new String[]{CAMERA}, REQUEST_CODE_CAMERA);
    }

    /**
     * Connects to the camera and sets up preview and analysis
     */
    void connectCamera() {
        ListenableFuture cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener((Runnable) () -> {
            // used to bind the lifecycle of cameras to the lifecycle owner
            ProcessCameraProvider cameraProvider  = null;
            try {
                cameraProvider = (ProcessCameraProvider) cameraProviderFuture.get();
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
            // preview
            Preview preview = new Preview.Builder()
                .build();
            preview.setSurfaceProvider(camera.getSurfaceProvider());
            ImageAnalysis imageAnalyzer = new ImageAnalysis.Builder()
                .build();
            imageAnalyzer.setAnalyzer(cameraExecutor, new ImageAnalyzer());
            // select back camera as a default
            CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
            try {
                // unbind use cases before rebinding
                cameraProvider.unbindAll();
                // bind use cases to camera
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer);
            } catch(Exception exc) {
                Log.e("TAG", "Use case binding failed", exc);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    /**
     * Releases resources when the activity is destroyed
     */
    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
    }

    /**
     * Opens the camera and sets up the UI for recognition
     */
    void openCamera(){
        isCameraOpened = true;
        btnSave.setVisibility(View.INVISIBLE);
        btnResume.setVisibility(View.INVISIBLE);
        imgPicked.setVisibility(View.INVISIBLE);
        resultView.setResult(null);
    }

    /**
     * Closes the camera and updates UI with the result
     */
    void closeCamera(){
        isCameraOpened = false;
        imgPicked.setVisibility(View.VISIBLE);
        btnSave.setVisibility(View.VISIBLE);
        btnResume.setVisibility(View.VISIBLE);
        resultView.setResult(currentResult);
        imgPicked.setImageBitmap(currentBitmap);
    }

    /**
     * Checks if all necessary permissions are granted
     */
    boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(getBaseContext(), CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * Handles the result of permission requests
     * @param requestCode  The request code passed in {@code requestPermissions(android.app.Activity, String[], int)}
     * @param permissions  The requested permissions, never null
     * @param grantResults The grant results for the corresponding permissions which is either
     *                     {@code PackageManager.PERMISSION_GRANTED} or {@code PackageManager.PERMISSION_DENIED}, never null
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        // handle the results for the camera permission request
        if (requestCode == REQUEST_CODE_CAMERA) {
            // if permissions are granted, set up and open the camera
            if (allPermissionsGranted()) {
                connectCamera();
                openCamera();
            } else {
                // if permissions are not granted, display a message and close the activity
                Toast.makeText(this, "Camera permissions not granted.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
        // handle the results for the storage permission request
        if(requestCode == Persistence.REQUEST_CODE_PERMISSIONS){
            if(Persistence.checkPermissions(this))
                btnSave.performClick(); // if permissions are granted, perform a click on the save button            
            else 
                Toast.makeText(this, "Storage permissions not granted, we can't save images.", Toast.LENGTH_SHORT).show();
        }
    }

    /**
     * Analyzes camera frames for real-time recognition
     */
    private class ImageAnalyzer implements ImageAnalysis.Analyzer {
        /**
         * Analyzes each camera frame to recognize objects
         *
         * @param image The frame to be analyzed
         */
        @Override
        public void analyze(@NonNull ImageProxy image) {
            runOnUiThread(() -> {
                // check if the camera is open; if not, release the image and return
                if(!isCameraOpened){
                    image.close();
                    return;
                }
                // retrieve the current frame as a Bitmap
                currentBitmap = camera.getBitmap();
                // if the bitmap is null, release the image and return
                if(currentBitmap == null){
                    image.close();
                    return;
                }
                // perform recognition on the current frame
    //            final Classifier.Recognition result = classifier.recognizeImageStrict(currentBitmap,camera.getDeviceRotationForRemoteDisplayMode());

                WindowManager windowManager = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
                int rotation = windowManager.getDefaultDisplay().getRotation();

                // Use 'rotation' as needed in your classifier
                final Classifier.Recognition result = classifier.recognizeImageStrict(currentBitmap, rotation);





                // if a result is obtained, update the UI with the recognition result
                if(result != null){
                    runOnUiThread(() -> {
                        // store the result and update the UI
                        currentResult = new RecognitionResultModel(result);
                        closeCamera();
                    });
                }
                image.close(); // release the image, to process the next frame
            });
        }
    }
// #endregion
}