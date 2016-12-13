package cz.pv021.neuralnets.functions;

/**
 *
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-13
 */
public class Softmax implements OutputFunction {
    @Override
    public double[] apply (double[] innerPotentials) {
        double[] output = new double[innerPotentials.length];
        double[] parts = new double[innerPotentials.length];
        double sumOfParts = 0;

        for (int i = 0; i < innerPotentials.length; i++) {
            parts[i] = Math.pow (Math.E, innerPotentials[i]);
        }
        
        for (int i = 0; i < parts.length; i++) {
            sumOfParts += parts[i];
        }
        
        for (int i = 0; i < parts.length; i++) {
            output[i] = parts[i] / sumOfParts;
        }
        
        return output;
    }

    @Override
    // Softmax_i' = out_i * (1 - out_i)
    public double[] derivative (double[] innerPotentials) {
        double[] apply = this.apply(innerPotentials);
        
        double[] output = new double[innerPotentials.length];
        for (int i=0; i<output.length; i++) {
            output[i] = apply[i] * (1 - apply[i]);
        }
        return output;
    }
}
