package cz.pv021.neuralnets.dataset;

import java.util.List;

/**
 * @param <C> Type of data class.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-12-15
 * @version 2016-12-15
 */
public interface SequenceExample <C extends DataClass> extends Example <List <double[]>, C> {
}
