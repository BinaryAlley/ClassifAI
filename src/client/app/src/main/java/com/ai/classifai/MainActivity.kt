package com.ai.classifai
// #region ================================================================== IMPORTS ====================================================================================
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.ai.classifai.helpers.Persistence
import org.tensorflow.lite.examples.classification.R
// #endregion

/**
 * Main activity of the application, serving as the primary interface
 *
 * Creation Date: 09th of January, 2021
 */
class MainActivity : AppCompatActivity(), View.OnClickListener {
// #region ================================================================== METHODS ====================================================================================
    /**
     * Initializes the activity and sets up listeners for UI elements
     *
     * @param savedInstanceState If the activity is re-initialized after previously being shut down, this contains the most recent data
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        // set click listeners for the buttons in the layout
        findViewById<View>(R.id.btn_rt_recog).setOnClickListener(this)
        findViewById<View>(R.id.btn_file_recog).setOnClickListener(this)
        findViewById<View>(R.id.btn_collection).setOnClickListener(this)
        findViewById<View>(R.id.btn_about).setOnClickListener(this)
        findViewById<View>(R.id.btn_quit).setOnClickListener(this)
        // check and request necessary permissions
        if (!Persistence.checkPermissions(this))
            Persistence.requestPermissions(this)
    }

    /**
     * Handles click events for all clickable views in the activity
     *
     * @param view The view that was clicked
     */
    override fun onClick(view: View) {
        // handle different button clicks
        when (view.id) {
            R.id.btn_rt_recog -> startActivity(RealtimeRecogActivity::class.java)
            R.id.btn_file_recog -> startActivity(FileRecogActivity::class.java)
            R.id.btn_collection -> startActivity(CollectionActivity::class.java)
            R.id.btn_about -> startActivity(AboutActivity::class.java)
            R.id.btn_quit -> finishAffinity()
        }
    }

    /**
     * Helper method to start a new activity
     *
     * @param destinationClass The class of the activity to start
     */
    private fun startActivity(destinationClass: Class<*>?) {
        startActivity(Intent(this, destinationClass))
    }

    /**
     * Callback for the result from requesting permissions
     *
     * @param requestCode  The request code passed in requestPermissions(android.app.Activity, String[], int)
     * @param permissions  The requested permissions
     * @param grantResults The grant results for the corresponding permissions
     */
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        // handle permissions result
        if (requestCode == Persistence.REQUEST_CODE_PERMISSIONS)
            if (!Persistence.checkPermissions(this))
                Toast.makeText(this, "Storage permissions not granted, we can't save images.", Toast.LENGTH_SHORT).show()
    }
// #endregion
}