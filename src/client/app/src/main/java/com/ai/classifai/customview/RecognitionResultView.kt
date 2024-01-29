package com.ai.classifai.customview
// #region ================================================================== IMPORTS ====================================================================================
import android.content.Context
import android.util.AttributeSet
import android.view.LayoutInflater
import android.widget.FrameLayout
import android.widget.TextView
import com.ai.classifai.models.RecognitionResultModel
import org.tensorflow.lite.examples.classification.R
// #endregion

/**
 * Custom view for displaying recognition results
 *
 * Creation Date: 09th of January, 2021
 */
class RecognitionResultView : FrameLayout {
// #region =============================================================== FIELD MEMBERS =================================================================================
    private var title: TextView? = null
    private var infoKingdom: InfoLabel? = null
    private var infoFamily: InfoLabel? = null
    private var infoSubFamily: InfoLabel? = null
    private var infoClade1: InfoLabel? = null
    private var infoClade2: InfoLabel? = null
    private var infoClade3: InfoLabel? = null
    private var infoOrder: InfoLabel? = null
    private var infoTribe: InfoLabel? = null
    private var infoGenus: InfoLabel? = null
// #endregion

// #region ================================================================= PROPERTIES ==================================================================================
    var result: RecognitionResultModel? = null
        set(result) {
            field = result
            if (result != null) {
                title!!.text = result.title
                infoKingdom!!.value = result.kingdom
                infoFamily!!.value = result.family
                infoSubFamily!!.value = result.subFamily
                infoClade1!!.value = result.clade1
                infoClade2!!.value = result.clade2
                infoClade3!!.value = result.clade3
                infoOrder!!.value = result.order
                infoTribe!!.value = result.tribe
                infoGenus!!.value = result.genus
            } else {
                title?.text = null
                infoKingdom!!.value = null
                infoFamily!!.value = null
                infoSubFamily!!.value = null
                infoClade1!!.value = null
                infoClade2!!.value = null
                infoClade3!!.value = null
                infoOrder!!.value = null
                infoTribe!!.value = null
                infoGenus!!.value = null
            }
        }

// #endregion

// #region ==================================================================== CTOR =====================================================================================
    constructor(context: Context?) : super(context!!) {
        init(null, 0)
    }

    constructor(context: Context?, attrs: AttributeSet?) : super(
        context!!, attrs
    ) {
        init(attrs, 0)
    }

    constructor(context: Context?, attrs: AttributeSet?, defStyle: Int) : super(
        context!!, attrs, defStyle
    ) {
        init(attrs, defStyle)
    }
// #endregion

// #region ================================================================== METHODS ====================================================================================
    /**
     * Initializes the view by inflating the layout and finding UI components
     * @param attrs The attributes of the XML tag that is inflating the view
     * @param defStyle The default style to apply to this view
     */
    private fun init(attrs: AttributeSet?, defStyle: Int) {
        // inflate the layout and find UI components
        val resultView = LayoutInflater
            .from(context)
            .inflate(R.layout.layout_recognition_result, this, true)
        title = resultView.findViewById(R.id.txt_title)
        infoKingdom = resultView.findViewById(R.id.info_kingdom)
        infoFamily = resultView.findViewById(R.id.info_family)
        infoSubFamily = resultView.findViewById(R.id.info_subfamily)
        infoClade1 = resultView.findViewById(R.id.info_clade1)
        infoClade2 = resultView.findViewById(R.id.info_clade2)
        infoClade3 = resultView.findViewById(R.id.info_clade3)
        infoOrder = resultView.findViewById(R.id.info_order)
        infoTribe = resultView.findViewById(R.id.info_tribe)
        infoGenus = resultView.findViewById(R.id.info_genus)
    }
// #endregion
}