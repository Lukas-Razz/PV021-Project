package cz.pv021.neuralnets.layers;

/**
 * @author  Josef Plch
 * @since   2016-11-21
 * @version 2016-11-21
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