package com.ai.classifai;

// #region ================================================================== IMPORTS ====================================================================================
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Bundle;
import android.widget.EditText;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import com.google.android.material.chip.Chip;
import com.squareup.picasso.Picasso;
import com.squareup.picasso.Target;
import com.ai.classifai.helpers.Persistence;
import com.ai.classifai.models.RecognitionResultModel;
import com.ai.classifai.tflite.Classifier;
import org.tensorflow.lite.examples.classification.R;
import com.ai.classifai.customview.RecognitionResultView;
import java.io.IOException;
// #endregion

/**
 * Activity for recognizing and processing image files.
 *
 * Creation Date: 09th of January, 2021
 */
public class FileRecogActivity extends AppCompatActivity {
// #region =============================================================== FIELD MEMBERS =================================================================================
    public static final int PICK_IMAGE = 1;
    private Classifier classifier;
    // views
    private EditText txt_path;
    private Chip btnOpenGallery, btnPredict, btnSave;
    private RecognitionResultView resultView;
    private Bitmap currentBitmap;
    private RecognitionResultModel currentResult;
// #endregion

// #region ================================================================== METHODS ====================================================================================
    /**
     * Initializes the activity, setting up UI components and event listeners
     *
     * @param savedInstanceState State of the application in a prior session
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_file_recog);
        // initialize UI components
        txt_path = findViewById(R.id.txt_path);
        btnOpenGallery = findViewById(R.id.btn_open_gallery);
        btnPredict = findViewById(R.id.btn_predict);
        btnSave = findViewById(R.id.btn_save);
        resultView = findViewById(R.id.result_view);
        // set up button click listeners
        btnSave.setOnClickListener(v -> {
            // save then
            if(currentBitmap != null && currentResult != null && Persistence.checkPermissions(this)){
                boolean saved = Persistence.self().savePrediction(currentBitmap, currentResult);
                if (saved){
                    Toast.makeText(this,"Prediction Saved.",Toast.LENGTH_SHORT).show();
                    clear();
                }
            }
            else {
                Persistence.requestPermissions(this);
            }
        });
        // open gallery button
        btnOpenGallery.setOnClickListener(v -> {
                // create an intent to pick an image from the gallery
                Intent intent = new Intent();
                intent.setType("image/*"); // set type to any image format
                intent.setAction(Intent.ACTION_GET_CONTENT); // get content from the device
                startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE);
            }
        );
        // predict button
        btnPredict.setOnClickListener(v -> {
                Uri uri = getImageUri(); // get image URI from text field
                if (uri != null && !uri.toString().isEmpty()){
                    // show loading dialog while processing
                    ProgressDialog pd = ProgressDialog.show(this, "Loading", "Wait while loading...");
                    // load image from URI and process it
                    Picasso.get().load(uri).into(new Target() {
                        @Override
                        public void onBitmapLoaded(Bitmap bitmap, Picasso.LoadedFrom from) {
                            // image successfully loaded
                            currentBitmap = bitmap;
                            // perform image recognition
                            final Classifier.Recognition result = classifier.recognizeImageStrict(bitmap, 90);
                            if(result != null){
                                // update UI with recognition result
                                currentResult = new RecognitionResultModel(result);
                                resultView.setResult(currentResult);
                                swtichSaveBtn(true); // enable save button
                                pd.dismiss(); // dismiss loading dialog
                            }
                            else{
                                // no result, prompt user to try another image
                                pd.dismiss();
                                Toast.makeText(getBaseContext(),"Couldn't find flowers, use another.",Toast.LENGTH_SHORT).show();
                                clear();
                            }
                        }
                        @Override
                        public void onBitmapFailed(Exception e, Drawable errorDrawable) {
                            // image loading failed
                            pd.dismiss();
                            Toast.makeText(getBaseContext(),"Couldn't load the image, use another.",Toast.LENGTH_SHORT).show();
                            clear(); // clear current data
                        }
                        @Override
                        public void onPrepareLoad(Drawable placeHolderDrawable) {
                            // additional code can be added here, if required before loading the image
                        }
                    });
                }
            }
        );
        // initialize classifier
        try {
            classifier = Classifier.create(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Clears the text field and result view, and disables the save button
     */
    private void clear(){
        txt_path.setText(null); // clear the text field
        resultView.setResult(null); // clear the result view
        swtichSaveBtn(false); // disable the save button
    }

    /**
     * Toggles the save button's clickability and background color
     *
     * @param on Boolean to enable (true) or disable (false) the save button
     */
    private void swtichSaveBtn(boolean on){
        if(on){
            btnSave.setClickable(true);
            btnSave.setChipBackgroundColor(ColorStateList.valueOf(getResources().getColor(R.color.color_accent)));
        }
        else {
            btnSave.setClickable(false);
            btnSave.setChipBackgroundColor(ColorStateList.valueOf(getResources().getColor(R.color.gray)));
        }

    }

    /**
     * Retrieves the image URI from the text field.
     *
     * @return Uri of the image.
     */
    private Uri getImageUri(){
        return Uri.parse(txt_path.getText().toString());
    }

    /**
     * Handles the result from the image selection activity
     *
     * @param requestCode The integer request code originally supplied to startActivityForResult(), allowing you to identify who this result came from
     * @param resultCode The integer result code returned by the child activity through its setResult()
     * @param data An Intent, which can return result data to the caller (various data can be attached to Intent "extras")
     */
    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode,resultCode,data);
        if (requestCode == PICK_IMAGE && resultCode == RESULT_OK) {
            if (data == null)
                // TODO: display an error
                return;
            else
                txt_path.setText(data.getData().toString()); // set the image URI to the text field
        }
    }

    /**
     * Callback for the result from requesting permissions
     *
     * @param requestCode The request code passed in requestPermissions(android.app.Activity, String[], int)
     * @param permissions The requested permissions, never null
     * @param grantResults The grant results for the corresponding permissions which is either PERMISSION_GRANTED or PERMISSION_DENIED, never null
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if(requestCode == Persistence.REQUEST_CODE_PERMISSIONS){
            if(Persistence.checkPermissions(this))
                btnSave.performClick(); // automatically click the save button if permissions are granted
            else
                Toast.makeText(this, "Storage permissions not granted, we can't save images.", Toast.LENGTH_SHORT).show();
        }
    }
// #endregion
}