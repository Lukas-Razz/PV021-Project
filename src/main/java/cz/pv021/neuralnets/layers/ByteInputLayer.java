package cz.pv021.neuralnets.layers;

import java.util.Arrays;

/**
 * @author  Josef Plch
 * @since   2016-11-21
 * @version 2016-12-12
 */
public class ByteInputLayer implements InputLayer {
    private static final int BYTE_SIZE = 8;
    private final InputLayer delegate = new InputLayerImpl (BYTE_SIZE);
    
    private static double boolToDouble (boolean bool) {
        return (bool == true) ? 1.0 : 0.0;
    }
    
    @Override
    public void setInput (double[] input) {
        this.delegate.setInput (input);
    }
    
    public static double[] byteToDoubleArray (byte byte8) {
        double[] array = new double[256];
        Arrays.fill (array, 0.0);
        int intValue = byte8 & 0xFF;
        array[intValue] = 1;
        return array;
    }
    
    private static boolean getBit (byte byte8, int bitIndex) {
        return (byte8 >> bitIndex & 1) == 1;
    }
    
    public void setInputByte (byte byte8) {
        double [] input = new double [BYTE_SIZE];
        for (int i = 0; i < BYTE_SIZE; i++) {
            input [i] = boolToDouble (getBit (byte8, i));
        }
        this.delegate.setInput (input);
    }
    
    @Override
    public LayerWithInput getLowerLayer () {
        return this.delegate.getLowerLayer ();
    }
    
    @Override
    public void setLowerLayer (LayerWithInput layer) {
        this.delegate.setLowerLayer (layer);
    }

    @Override
    public int getNumberOfUnits () {
        return this.delegate.getNumberOfUnits ();
    }

    @Override
    public double[] getOutput () {
        return this.delegate.getOutput ();
    }
}
