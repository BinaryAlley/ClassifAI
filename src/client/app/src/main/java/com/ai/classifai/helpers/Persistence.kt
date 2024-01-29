package com.ai.classifai.helpers
// #region ================================================================== IMPORTS ====================================================================================
import android.Manifest.permission
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Environment
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.ai.classifai.models.RecognitionResultModel
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Arrays
import java.util.Date
import java.util.Locale
// #endregion

/**
 * Utility class for managing file persistence and permissions
 *
 * Creation Date: 09th of January, 2021
 */
class Persistence private constructor() {
    // #region =============================================================== FIELD MEMBERS =================================================================================
    private val folder: File = File(Environment.getExternalStorageDirectory().toString() + File.separator + "Collection")

// #endregion

// #region ==================================================================== CTOR =====================================================================================
    init {
    var success = true
        if (!folder.exists()) success = folder.mkdirs()
    }

    /**
     * Deletes a file at the specified path
     *
     * @param path The path of the file to be deleted
     * @return True if the file is deleted successfully, false otherwise
     */
    fun delete(path: String?): Boolean {
        val imgFile = File(path)
        return if (imgFile.exists()) imgFile.delete() else false
    }

    val collectionFilesPaths: Array<String?>
        /**
         * Retrieves paths of files in the collection folder, sorted by modification date
         *
         * @return An array of file paths in the collection folder
         */
        get() {
            // get a list of files in the collection folder
            val files = folder.listFiles()
            // if there are no files or the folder is empty, return an empty array
            if (files == null || files.isEmpty()) return arrayOf()
            // sort the files based on their last modification date in descending order
            Arrays.sort(files) { o1, o2 -> (o2!!.lastModified() - o1!!.lastModified()).toInt() }
            // create an array to store the file paths
            val paths = arrayOfNulls<String>(files.size)
            // generate absolute paths for each file and store them in the array
            for (index in files.indices) paths[index] =
                folder.absolutePath + File.separator + files[index].name
            return paths
        }

    /**
     * Saves a Bitmap image with a specified title to external storage
     *
     * @param bitmap The Bitmap image to be saved
     * @param title  The title used for generating the filename
     * @return True if the image is saved successfully, false otherwise
     */
    private fun saveBitmap(bitmap: Bitmap, title: String): Boolean {
        // generate a timestamp for the filename
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val filename = title.lowercase(Locale.getDefault()) + "_" + timeStamp + ".png"
        if (folder.exists()) {
            // create a File object representing the destination file
            val file = File(folder.absolutePath, filename)
            try {
                FileOutputStream(file).use { out ->
                    val success =
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                    // PNG is a lossless format, the compression factor (100) is ignored
                    out.flush()
                    out.close()
                    //scanFile(file.getAbsolutePath());
                    return success
                }
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
        return false
    }

    /**
     * Saves a prediction image to external storage
     *
     * @param bitmap The bitmap image to be saved
     * @param model  The recognition result model associated with the image
     * @return True if the image is saved successfully, false otherwise
     */
    fun savePrediction(bitmap: Bitmap, model: RecognitionResultModel): Boolean {
        return if (isExternalStorageWritable) saveBitmap(bitmap, model.title) else false
    }

    private val isExternalStorageWritable: Boolean
        /**
         * Checks if external storage is writable
         *
         * @return True if external storage is mounted and writable, false otherwise
         */
        get() {
            // get the current state of external storage
            val state = Environment.getExternalStorageState()
            // check if external storage is mounted and writable
            return Environment.MEDIA_MOUNTED == state
        }

    companion object {
        private var persistence: Persistence? = null
        const val REQUEST_CODE_PERMISSIONS = 820
// #endregion

// #region ================================================================== METHODS ====================================================================================
        /**
         * Requests necessary permissions for reading and writing external storage
         *
         * @param activity The activity from which the permissions are requested
         */
        @JvmStatic
        fun requestPermissions(activity: Activity?) {
            ActivityCompat.requestPermissions(
                activity!!,
                arrayOf(permission.READ_EXTERNAL_STORAGE, permission.WRITE_EXTERNAL_STORAGE),
                REQUEST_CODE_PERMISSIONS
            )
        }

        /**
         * Gets the instance of the Persistence class.
         *
         * @return The Persistence instance.
         */
        @JvmStatic
        fun self(): Persistence? {
            return persistence
        }

        /**
         * Initializes the Persistence class
         *
         * @param context The application context
         */
        fun init(context: Context?) {
            checkPermissions(context)
            if (persistence == null) persistence = Persistence()
        }

        /**
         * Checks if necessary permissions for reading and writing external storage are granted
         *
         * @param context The application context
         * @return True if permissions are granted, false otherwise
         */
        @JvmStatic
        fun checkPermissions(context: Context?): Boolean {
            return ContextCompat.checkSelfPermission(
                context!!,
                permission.WRITE_EXTERNAL_STORAGE
            ) == PackageManager.PERMISSION_GRANTED &&
                    ContextCompat.checkSelfPermission(
                        context,
                        permission.READ_EXTERNAL_STORAGE
                    ) == PackageManager.PERMISSION_GRANTED
        }
// #endregion
    }
}