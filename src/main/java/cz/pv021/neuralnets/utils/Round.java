package cz.pv021.neuralnets.utils;

/**
 *
 * @author lukas
 */
public class Round {
    
    public static double Round(double value, double by) {
        return Math.round(value*by)/by;
    }
    
    public static double[] Round(double[] values, double by) {
        double[] newValues = new double[values.length];
        for(int i=0; i<newValues.length; i++) {
            newValues[i] = Math.round(values[i]*by)/by;
        }
        return newValues;
    }
}
