package cz.pv021.neuralnets.initialization;

import cz.pv021.neuralnets.utils.LayerParameters;

/**
 * Interface for initialization functions
 * 
 * @author  Lukáš Daubner
 * @since   2016-12-10
 * @version 2016-12-14
 */
public interface Initialization {

    LayerParameters initializeSIGMOIDLike(LayerParameters parameters);
    
    LayerParameters initializeTANHLike(LayerParameters parameters);
    
}
