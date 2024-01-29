package com.ai.classifai
// #region ================================================================== IMPORTS ====================================================================================
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.examples.classification.R
// #endregion

/**
 * Activity for displaying the About page of the application.
 *
 * Creation Date: 09th of January, 2021
 */
class AboutActivity : AppCompatActivity() {
// #region ================================================================== METHODS ====================================================================================
    /**
     * Sets up the about page layout.
     *
     * @param savedInstanceState State of the application in a prior session.
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_about)
    }
// #endregion
}
