package com.ai.classifai.customview
// #region ================================================================== IMPORTS ====================================================================================
import android.content.Context
import android.util.AttributeSet
import androidx.appcompat.widget.AppCompatImageView
// #endregion

/**
 * Custom ImageView that enforces a square aspect ratio
 *
 * Creation Date: 09th of January, 2021
 */
class SquareImageView : AppCompatImageView {
// #region ==================================================================== CTOR =====================================================================================
    constructor(context: Context?) : super(context!!)
    constructor(context: Context?, attrs: AttributeSet?) : super(
        context!!, attrs
    )

    constructor(context: Context?, attrs: AttributeSet?, defStyleAttr: Int) : super(
        context!!, attrs, defStyleAttr
    )
// #endregion

// #region ================================================================== METHODS ====================================================================================
    /**
     * Measure the view and enforce a square aspect ratio
     *
     * @param widthMeasureSpec  Width measurement specification
     * @param heightMeasureSpec Height measurement specification
     */
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec)
        val width = measuredWidth
        setMeasuredDimension(width, width)
    }
// #endregion
}