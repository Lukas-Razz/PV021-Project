package cz.pv021.neuralnets.layers;

/**
 * @author  Josef Plch
 * @since   2016-12-12
 * @version 2016-12-14
 */
public interface RecursiveHiddenLayer extends HiddenLayer {
    public HiddenLayer feedForwardCopy ();
    
    /*
    public default void unfold (int k) {
        List <LayerWithOutput> previousLayers = this.getInputLayers ();
        for (int i = 0; i < k; i++) {
            HiddenLayer clone = this.feedForwardCopy ();
            Layers.connect (previousLayers, clone);
            previousLayers = Collections.singletonList (clone);
        }
        Layers.connect (previousLayers, this.getOutputLayer ());
    }
    */
}
