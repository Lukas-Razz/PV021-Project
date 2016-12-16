package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.utils.ByteUtils;

/**
 * @author  Josef Plch
 * @since   2016-11-21
 * @version 2016-12-15
 */
public class ByteInputLayer extends InputLayerImpl {
    private static final int SIZE = 256;

    public ByteInputLayer (int id) {
        super (id, SIZE);
    }
    
    @Override
    public ByteInputLayer deepCopy (int id) {
        return (new ByteInputLayer (id));
    }
    
    public void setInput (byte byte8) {
        this.setInput (ByteUtils.byteToOneHotVector (byte8));
    }
}
