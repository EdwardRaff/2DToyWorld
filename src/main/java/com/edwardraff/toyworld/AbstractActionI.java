/*
 * Copyright (C) 2014 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.edwardraff.toyworld;

import java.awt.event.ActionEvent;
import javax.swing.AbstractAction;

/**
 * Convenience interface used to make theTransformsMenu code a little cleaner
 * @author Edward Raff
 */
public interface AbstractActionI
{
    public void actionPerformedI(ActionEvent e);
    
    
    default public AbstractAction getAsAction()
    {
        return new AbstractAction()
        {

            @Override
            public void actionPerformed(ActionEvent e)
            {
                actionPerformedI(e);
            }
        };
    }
}
