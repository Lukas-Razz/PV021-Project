package cz.pv021.neuralnets.layers;

/**
 *
 * @author Josef Plch
 */
public interface RecursiveHiddenLayer extends HiddenLayer {
    public HiddenLayer feedForwardCopy ();
    
    public default void unfold (int k) {
        LayerWithOutput previousLayer = this.getUpperLayer ();
        for (int i = 0; i < k; i++) {
            HiddenLayer clone = this.feedForwardCopy ();
            Layers.connect (previousLayer, clone);
            previousLayer = clone;
        }
        Layers.connect (previousLayer, this.getLowerLayer ());
    }    
}
