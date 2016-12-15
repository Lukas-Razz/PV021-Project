package cz.pv021.neuralnets.dataset.udcorpora;

import cz.pv021.neuralnets.dataset.DataClass;

/**
 * @author  Josef Plch
 * @since   2016-12-15
 * @version 2016-12-15
 */
public enum UdLanguage implements DataClass {
    CZECH ("cs"),
    ENGLISH ("en"),
    FRENCH ("fr"),
    GERMAN ("de"),
    HUNGARIAN ("hu"),
    ITALIAN ("it"),
    LATIN ("la"),
    POLISH ("pl"),
    SPANISH ("es");
    private final String code;
    
    private UdLanguage (String code) {
        this.code = code;
    }
    
    public String getCode () {
        return code;
    }
    
    @Override
    public int getIndex () {
        return this.ordinal ();
    }

    @Override
    public String getName() {
        return this.name ();
    }
    
    public static int size () {
        return values().length;
    }
}
