package cz.pv021.neuralnets.utils;

import cz.pv021.neuralnets.dataset.DataClass;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;

/**
 * @author  Josef Plch
 * @since   2016-12-13
 * @version 2016-12-16
 */
public abstract class ModelStatistics {
    private static final DecimalFormat DEFAULT_DECIMAL_FORMAT = createDefaultDecimalFormat ();
    
    public static DecimalFormat createDefaultDecimalFormat () {
        DecimalFormat format = new DecimalFormat ("0.0");
        DecimalFormatSymbols formatSymbols = new  DecimalFormatSymbols ();
        formatSymbols.setDecimalSeparator ('.');
        format.setDecimalFormatSymbols (formatSymbols);
        return format;
    }
    
    public static String modelStatistics (int[][] confusionMatrix, DataClass[] classes) {
        return modelStatistics (confusionMatrix, classes, DEFAULT_DECIMAL_FORMAT);
    }
    
    /**
     * Get model statistics (serialized to a string).
     * 
     * @param confusionMatrix It must be square matrix.
     * @param classesx        List of classes ordered by their indices.
     * @param formatter       Formatter of decimal numbers.
     * @return                Serialized statistics.
     */
    public static String modelStatistics (int[][] confusionMatrix, DataClass[] classesx, DecimalFormat formatter) {
        int classes = confusionMatrix.length;
        
        int totalInstances = 0;
        int correct = 0;
        for (int x = 0; x < classes; x++) {
            correct += confusionMatrix[x][x];
            for (int y = 0; y < classes; y++) {
                totalInstances += confusionMatrix[x][y];
            }
        }
        double overallAccuracy = 100.0 * correct / totalInstances;
        
        // Overall statistics.
        StringBuilder result = new StringBuilder ();
        result
            .append ("Correctly classified instances  \t")
            .append (correct).append("\t").append(formatter.format (overallAccuracy)).append (" %")
            .append ("\nIncorrectly classified instances\t")
            .append (totalInstances - correct).append("\t").append(formatter.format (100 - overallAccuracy)).append (" %")
            .append ("\nTotal number of instances       \t")
            .append (totalInstances)
            .append ("\n");
        
        // Confusion matrix heading.
        result.append ("\nPredicted ->");
        for (int x = 0; x < classes; x++) {
            result.append ("\t").append (x);
        }
        result.append ("\tPrec.").append ("\tRecall");
        
        result.append ("\n--------------------------------------------------------");
        
        // Confusion matrix body.
        for (int x = 0; x < classes; x++) {
            // Class index: class name
            result.append ("\n").append (x).append (": ").append (classesx[x].getName ());
            
            int xAsX = confusionMatrix[x][x];
            int xAsAny = 0;
            int anyAsX = 0;
            for (int y = 0; y < classes; y++) {
                int xAsY = confusionMatrix[x][y];
                int yAsX = confusionMatrix[y][x];
                xAsAny += xAsY;
                anyAsX += yAsX;
                result.append ("\t").append (xAsY);
            }
            
            // Precision = TP / (TP + FP)
            double classPrecision = 100.0 * xAsX / anyAsX;
            result.append ("\t").append (formatter.format (classPrecision));
            
            // Recall = TP / (TP + FN)
            double classRecall = 100.0 * xAsX / xAsAny;
            result.append("\t").append (formatter.format (classRecall));
        }
        
        return result.toString ();
    }
    
    public static String show2dArray (double[][] array) {
        return show2dArray (array, DEFAULT_DECIMAL_FORMAT);
    }
    
    public static String show2dArray (double[][] array, DecimalFormat decimalFormat) {
        StringBuilder result = new StringBuilder ();
        for (double[] row : array) {
            for (double x: row) {
                result.append (decimalFormat.format (x));
                result.append ('\t');
            }
            result.append ('\n');
        }
        return result.toString ();
    }
}
