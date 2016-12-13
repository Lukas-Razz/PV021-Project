package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.utils.ByteUtils;

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
    
    public void setInputByte (byte byte8) {
        double [] input = new double [BYTE_SIZE];
        for (int i = 0; i < BYTE_SIZE; i++) {
            input [i] = boolToDouble (ByteUtils.getBit (byte8, i));
        }
        this.delegate.setInput (input);
    }
    
    @Override
    public LayerWithInput getOutputLayer () {
        return this.delegate.getOutputLayer ();
    }
    
    @Override
    public void setOutputLayer (LayerWithInput layer) {
        this.delegate.setOutputLayer (layer);
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
