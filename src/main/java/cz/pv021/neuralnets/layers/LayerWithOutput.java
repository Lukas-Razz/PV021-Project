package cz.pv021.neuralnets.layers;

/**
 * Any layer with following layer, i.e. any except the output layer.
 * 
 * @author  Josef Plch
 * @since   2016-11-08
 * @version 2016-11-08
 */
public interface LayerWithOutput extends Layer {
    /**
     * Nastav navazující (výstupní) vrstvu.
     * 
     * @param layer 
     */
    public void setLowerLayer (LayerWithInput layer);
}
