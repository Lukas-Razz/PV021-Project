package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.utils.ByteUtils;

/**
 * @author  Josef Plch
 * @since   2016-11-21
 * @version 2016-12-13
 */
public class ByteInputLayer implements InputLayer <Byte> {
    private static final int SIZE = 256;
    private final InputLayer delegate = new InputLayerImpl (SIZE);
    
    @Override
    public int getNumberOfUnits () {
        return SIZE;
    }

    @Override
    public double[] getOutput () {
        return this.delegate.getOutput ();
    }
    
    @Override
    public LayerWithInput getOutputLayer () {
        return this.delegate.getOutputLayer ();
    }
    
    @Override
    public void setInput (double[] input) {
        this.delegate.setInput (input);
    }
    
    @Override
    public void setInputObject (Byte byte8) {
        this.setInput (ByteUtils.byteToOneHotVector (byte8));
    }
    
    @Override
    public void setOutputLayer (LayerWithInput layer) {
        this.delegate.setOutputLayer (layer);
    }
}
