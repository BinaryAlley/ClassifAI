package com.ai.classifai.customview
// #region ================================================================== IMPORTS ====================================================================================
import android.content.Context
import android.util.AttributeSet
import android.widget.LinearLayout
import android.widget.TextView
import org.tensorflow.lite.examples.classification.R

// #endregion
/**
 * Custom Control for displaying a label and a value
 *
 * Creation Date: 10th of January, 2021
 */
internal class InfoLabel : LinearLayout {
// #region =============================================================== FIELD MEMBERS =================================================================================
    // Views
    private var tv_property: TextView? = null
    private var tv_value: TextView? = null

// #endregion

// #region ================================================================= PROPERTIES ==================================================================================
    // Values
    private var property: String? = null
        set(value) {
            var str = value
            if (str!![str.length - 1] != ':') str += ":"
            if (tv_property != null) tv_property!!.text = str
            field = str
            invalidate()
            requestLayout()
        }
    var value: String? = null
        set(value) {
            field = value
            if (tv_value != null) {
                tv_value!!.text = value
            }
            invalidate()
            requestLayout()
        }

// #endregion

// #region ==================================================================== CTOR =====================================================================================
    constructor(context: Context?) : super(context) {
        init(null, 0)
    }

    constructor(context: Context?, attrs: AttributeSet?) : super(context, attrs) {
        init(attrs, 0)
    }

    constructor(context: Context?, attrs: AttributeSet?, defStyle: Int) : super(
        context,
        attrs,
        defStyle
    ) {
        init(attrs, defStyle)
    }
//#endregion

// #region ================================================================== METHODS ====================================================================================
    /**
     * Initializes the control with given attributes; constructs TextViews and sets their properties
     * @param attrs The attributes for the control
     * @param defStyle The default style to apply to this view
     */
    private fun init(attrs: AttributeSet?, defStyle: Int) {
        // building
        val layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT, 1f)
        tv_property = TextView(context)
        tv_property?.setTextAppearance(R.style.InfoLabelPropertyText)

        tv_property!!.layoutParams = layoutParams
        tv_value = TextView(context)
        tv_value?.setTextAppearance(R.style.InfoLabelValueText)
        tv_value!!.textAlignment = TEXT_ALIGNMENT_TEXT_END
        tv_value!!.layoutParams = layoutParams
        weightSum = 2f
        this.orientation = HORIZONTAL
        this.addView(tv_property)
        this.addView(tv_value)
        // initialize the control's properties from XML attributes
        val ta = context.theme.obtainStyledAttributes(attrs, R.styleable.InfoLabel, 0, 0)
        try {
            property = ta.getString(R.styleable.InfoLabel_property)
            value = ta.getString(R.styleable.InfoLabel_value)
        } finally {
            ta.recycle()
        }
    }
// #endregion
}