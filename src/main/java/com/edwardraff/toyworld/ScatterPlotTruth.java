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

import java.awt.*;
import jsat.graphing.ProgressPanel;
import jsat.graphing.ScatterPlot;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.regression.Regressor;

/**
 *
 * @author Edward Raff
 */
public class ScatterPlotTruth extends ScatterPlot
{
    private final Function truth;

    public ScatterPlotTruth(Vec xValues, Vec yValues, Regressor regressor, Function truth)
    {
        super(xValues, yValues, regressor);
        this.truth = truth;
    }

    @Override
    protected void paintWork(Graphics g, int imageWidth, int imageHeight, ProgressPanel pp)
    {
        super.paintWork(g, imageWidth, imageHeight, pp);
        
        if(truth != null)
        {
            g.setColor(Color.GREEN);
            drawFunction((Graphics2D)g, truth);
        }
    }

    
    
}
