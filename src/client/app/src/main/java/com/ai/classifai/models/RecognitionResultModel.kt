package com.ai.classifai.models

import com.ai.classifai.tflite.Classifier.Recognition
import java.text.DecimalFormat
import java.util.Locale

// #region ================================================================== IMPORTS ====================================================================================
// #endregion
/**
 * Represents a model for recognition results, including label, percentage, and
 * taxonomy information
 *
 * Creation Date: 09th of January, 2021
 */
class RecognitionResultModel(recognition: Recognition) {
// #region =============================================================== FIELD MEMBERS =================================================================================
    private var df = DecimalFormat("(#.##%)")
    private var label: String
    private var percentage: Double
    lateinit var kingdom: String
    lateinit var family: String
    lateinit var subFamily: String
    lateinit var clade1: String
    lateinit var clade2: String
    lateinit var clade3: String
    lateinit var order: String
    lateinit var tribe: String
    lateinit var genus: String
    val title: String
        get() {
            val firstLetter = label[0]
            var result = if (!Character.isUpperCase(firstLetter)) {
                firstLetter.uppercase(Locale.getDefault()) + label.substring(1)
            } else {
                label
            }
            result += " $percentageStr"
            return result
        }
    private val percentageStr: String
        get() = df.format(percentage)
// #endregion

// #region ==================================================================== CTOR =====================================================================================
    /**
     * Initializes a new RecognitionResultModel based on a given Classifier.Recognition instance
     *
     * @param recognition The Classifier.Recognition instance
     */
    init {
        label = recognition.title
        percentage = recognition.confidence.toDouble()
        when (label) {
            "daisy" -> {
                kingdom = "Plantae"
                clade1 = "Tracheophytes"
                clade2 = "Angiosperms"
                clade3 = "Eudicots"
                order = "Asterales"
                family = "Asteraceae"
                subFamily = "Asteroideae"
                tribe = "Astereae"
                genus = "Bellis"
            }

            "dandelion" -> {
                kingdom = "Plantae"
                clade1 = "Tracheophytes"
                clade2 = "Angiosperms"
                clade3 = "Eudicots"
                order = "Asterales"
                family = "Asteraceae"
                subFamily = "Cichorioideae"
                tribe = "Cichorieae"
                genus = "Taraxacum"
            }

            "roses" -> {
                kingdom = "Plantae"
                clade1 = "Tracheophytes"
                clade2 = "Angiosperms"
                clade3 = "Eudicots"
                order = "Rosales"
                family = "Rosaceae"
                subFamily = "Rosoideae"
                tribe = "Roseae"
                genus = "Rosa"
            }

            "sunflowers" -> {
                kingdom = "Plantae"
                clade1 = "Tracheophytes"
                clade2 = "Angiosperms"
                clade3 = "Eudicots"
                order = "Asterales"
                family = "Asteraceae"
                subFamily = "Asteroideae"
                tribe = "Heliantheae"
                genus = "Helianthus"
            }

            "tulips" -> {
                kingdom = "Plantae"
                clade1 = "Tracheophytes"
                clade2 = "Angiosperms"
                clade3 = "Monocots"
                order = "Liliales"
                family = "Liliaceae"
                subFamily = "Lilioideae"
                tribe = "Lilieae"
                genus = "Tulipa"
            }
        }
    }
// #endregion
}