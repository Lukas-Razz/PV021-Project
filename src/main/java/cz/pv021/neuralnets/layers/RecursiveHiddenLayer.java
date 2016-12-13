package cz.pv021.neuralnets.layers;

/**
 * @author  Josef Plch
 * @since   2016-12-12
 * @version 2016-12-13
 */
public interface RecursiveHiddenLayer extends HiddenLayer {
    public HiddenLayer feedForwardCopy ();
    
    public default void unfold (int k) {
        LayerWithOutput previousLayer = this.getInputLayer ();
        for (int i = 0; i < k; i++) {
            HiddenLayer clone = this.feedForwardCopy ();
            Layers.connect (previousLayer, clone);
            previousLayer = clone;
        }
        Layers.connect (previousLayer, this.getOutputLayer ());
    }    
}
