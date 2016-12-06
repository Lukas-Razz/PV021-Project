package cz.pv021.neuralnets.functions;

/**
 *
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-06
 */
public class Softmax implements OutputFunction {
    @Override
    public double[] apply (double[] innerPotencials) {
        double[] output = new double[innerPotencials.length];
        double[] parts = new double[innerPotencials.length];
        double sumOfParts = 0;

        for (int i = 0; i < innerPotencials.length; i++) {
            parts[i] = Math.pow (Math.E, innerPotencials[i]);
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
    // Softmax_i' = out_i * ( 1 - out_i )
    public double[] derivative (double[] innerPotencials) {
        double[] apply = this.apply(innerPotencials);
        
        double[] output = new double[innerPotencials.length];
        for(int i=0; i<output.length; i++)
        {
            output[i] = apply[i] * ( 1 - apply[i] );
        }
        return output;
    }
}
