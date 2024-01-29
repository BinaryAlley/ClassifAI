package com.ai.classifai

import android.app.Application
import android.os.StrictMode
import com.ai.classifai.helpers.Persistence

/**
 * Main application class
 * Initializes application-wide settings and configurations
 *
 * Creation Date: 09th of January, 2021
 */
class App : Application() {
    /**
     * Called when the application is starting, before any other application objects have been created
     * Used to configure settings and initialize state
     */
    override fun onCreate() {
        super.onCreate()
        // set up the StrictMode policy for the application
        val builder = StrictMode.VmPolicy.Builder()
        StrictMode.setVmPolicy(builder.build())
        Persistence.checkPermissions(this)
        Persistence.init(this)
    }
}