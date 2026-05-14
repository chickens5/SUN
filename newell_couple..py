#UMSL ~ Computing Technology
#Gabriel J 
#May 13 2026


#Welcome, I made this personal project so anyone can pick it up and make space weather predictions.

    # Our understanding of space weather is microscopic equal to our place in it's vastness.
    #The effects of space weather are understudied and often underestimated, but they can have significant impacts on our technology and daily lives.
    #The industrilization of AI and World conflicts may have unknown affects on space weather and the Earth's various systems, but alas, I digress...


# **Key Concepts:**

# The Newell Coupling function is NOT universally applicable to all space weather problems (e.g., solar flare prediction, Coronal Mass Ejection forecasting).

    #The physical model the function is based on is specific to the interplaneteary magnetic field and solar wind interactions with the Earth's magnetosphhere.

    #Therefore, it is accurate for predicting geomagnetic activity (Kp index) and related phenomena that are directly influenced by solar wind-magnetosphere coupling.


        # # **The Newell Coupling Function** = parameter estimates
        # the rate of magnetic energy transfer at the dayside magnetopause based on solar wind properties
        # (dynamic pressure, magnetic field orientation, and speed). This energy transfer drives
        # geomagnetic disturbances and is crucial for understanding and predicting geomagnetic storms.

        # **Magnetosphere** = the region of space around Earth dominated by Earth's magnetic field,
        # where Earth's magnetic force is stronger than the interplanetary magnetic field from the Sun.
        # It acts as a protective bubble around the planet.

        # **Solar Wind** = a continuous stream of charged particles (plasma) flowing outward from the Sun
        # throughout the solar system. It carries an embedded magnetic field (the interplanetary magnetic field)
        # and continuously interacts with planetary magnetospheres.

# **What does the Newell Coupling parameter quantify?**

    # The Newell coupling parameter provides a quantitative measure of the efficiency of energy
    # transfer from the solar wind into Earth's magnetosphere. It combines three key solar wind
    # properties: ram pressure, magnetic field strength, and the southward component of the
    # interplanetary magnetic field (IMF Bz).

# **How is it used for prediction?**

    # Since the Newell parameter directly correlates with geomagnetic activity (Kp index),
    # we can use real-time or near-real-time solar wind measurements to estimate the likelihood
    # and intensity of geomagnetic storms. However, this works specifically for geomagnetic prediction—
    # it does NOT predict solar flares or other space weather phenomena without additional physics models.



#``````````

  #Although Carrington-like events are rare, Solar flares and Coronal Mass Ejections from the Sun can cause significant disruptions to Earth's magnetosphere, 
    # leading to geomagnetic storms that can affect satellite operations, power grids, and communication systems.

    # There also may be a correlation between solar activity and climate patterns on Earth, although this is an area of ongoing research. 

    # By understanding and predicting space weather, 
    # we can better prepare for and mitigate the impacts of solar events given the World's current technological dependence and neglect for the environment.

#  Okay I stop ranting now!!!! ~ Gabe

#``````````

#The GGSP Jupyter Notebook along with the following Python script utilize the Newell Coupling function for Solar Wind-Magnetosphere interactions. 

# The paper it's based on can be found here:  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JA012015 
# and here (University of Reading PDF): chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://centaur.reading.ac.uk/102391/9/Space%20Weather%20-%202022%20-%20Lockwood%20-%20Solar%20Wind%20Magnetosphere%20Coupling%20Functions%20%20Pitfalls%20%20Limitations%20%20and%20Applications.pdf


#LETS GET TO THE GOOD STUFF

import tkinter as tk
from tkinter import ttk

import numpy as np  #need numpy for physics and calculations.


#This function computes the paramteter in which Earth's magnetosphere will be affected by Solar Wind and the Interplanetary Magnetic Field (IMF) --via coupling Functions-- 
    
    #The formula (Newell et al., 2007 proxy) is as followS:

        #dPhi / dt 

            #dPhi / dt = the rate of magnetic flux opening at Earth's magnetopause. 


    # dPhi / dt = V^(4/3) * Bt^(2/3) * sin(theta_c / 2)^(8/3)

    #Dividing the rate of magnetic flux opening at Earth's magnetopause by the solar wind parameters can predict the geomagnetic activity level (Kp index) 
    # and related phenomena that are directly influenced by solar wind-magnetosphere coupling.
   
    #The necessary variables are as follows:
        #V is the Solar Wind speed in km/s
        #Bt = sqrt(By^2 + Bz^2) is the transverse IMF magnitude in nT
        #theta_c is the IMF clock angle in radians (computed from By and Bz)

    #This implementation is NumPy-based and accepts scalars or arrays.


def newells_coupling_function(v_kms, by_nt, bz_nt):

    #The function takes in the following (represents variables above):
        #v_kms: Solar wind speed in km/s.
        #by_nt: IMF By component in nT (GSM-like interpretation).
        #bz_nt: IMF Bz component in nT.


    # We must convert speed, by, and bz into arrays (with np.asarray)so scalar and vector use-cases share the same path.

    speed = np.asarray(v_kms, dtype=float)  # Solar wind speed in km/s.
    by = np.asarray(by_nt, dtype=float) # IMF By component in nT (GSM-like interpretation).
    bz = np.asarray(bz_nt, dtype=float) # IMF Bz component in nT.

    # Bt captures IMF strength in the plane transverse to the Sun-Earth line.
    bt_nt = np.sqrt(by**2 + bz**2)

    # Clock angle uses |By| to keep theta in [0, pi] for this proxy convention.
    clock_angle_rad = np.arctan2(np.abs(by), bz)

    # Newell coupling proxy dPhi/dt.
    coupling = (
        np.power(speed, 4.0 / 3.0)  #Utilizes np.power to compute the speed raised to the 4/3 power.
        * np.power(bt_nt, 2.0 / 3.0)    #Utilizes np.power to compute the transverse IMF magnitude raised to the 2/3 power.
        * np.power(np.sin(clock_angle_rad / 2.0), 8.0 / 3.0)    #Utilizes np.power to compute the sine of half the clock angle raised to the 8/3 power.
    )

    clock_angle_deg = np.degrees(clock_angle_rad)   #Assigns the clock angle in degrees by converting the clock angle from radians using np.degrees.
    return coupling, bt_nt, clock_angle_deg     #Returns the computed coupling parameter, transverse IMF magnitude, and clock angle in degrees as a tuple.


# Input ranges chosen for practical near-Earth solar wind/IMF exploration in a GUI.
ACCEPTABLE_RANGES = {
    "v_kms": (200.0, 1200.0),
    "by_nt": (-80.0, 80.0),
    "bz_nt": (-80.0, 80.0),
}


def parse_input_with_range(name, raw_text):

    #Method to parse and validate user input with our predefined acceptable ranges for solar wind speed and IMF components.
    #  It raises ValueError with informative messages if the input is invalid or out of range.
   
    min_val, max_val = ACCEPTABLE_RANGES[name] #Assigns our min/max vals to acceptable_ranges && [name] for given input name (e.g., "v_kms", "by_nt", "bz_nt").
    try:
        value = float(raw_text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc

    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}.")

    return value


class NewellCouplingGUI(tk.Tk):
    """Small desktop UI to explore Newell coupling with bounded user input."""

    def __init__(self):
        super().__init__()
        self.title("Newell Coupling Function Explorer")
        self.geometry("980x620")
        self.resizable(False, False)

        # Entry variables store user input strings before validation/conversion.
        self.v_var = tk.StringVar(value="450")
        self.by_var = tk.StringVar(value="5")
        self.bz_var = tk.StringVar(value="-8")

        # Result variables drive output labels after each computation.
        self.bt_var = tk.StringVar(value="-")
        self.theta_var = tk.StringVar(value="-")
        self.coupling_var = tk.StringVar(value="-")
        self.status_var = tk.StringVar(value="Enter values and click Compute.")

        # Canvas dimensions for a simple coupling-rate curve visualization.
        self.plot_width = 520
        self.plot_height = 300
        self.plot_padding = 40
        self.plot_canvas = None

        self._build_layout()
        self._on_compute()

    def _build_layout(self):
        """Create and place all GUI widgets."""
        pad = {"padx": 10, "pady": 6}

        title = ttk.Label(
            self,
            text="Newell Coupling Function (Earth-Sun Solar Wind Inputs)",
            font=("Segoe UI", 13, "bold"),
        )
        title.grid(row=0, column=0, columnspan=3, sticky="w", **pad)

        constraints_frame = ttk.LabelFrame(self, text="Predefined Variable Constraints")
        constraints_frame.grid(row=1, column=0, columnspan=3, sticky="we", padx=10, pady=6)

        v_min, v_max = ACCEPTABLE_RANGES["v_kms"]
        by_min, by_max = ACCEPTABLE_RANGES["by_nt"]
        bz_min, bz_max = ACCEPTABLE_RANGES["bz_nt"]

        ttk.Label(constraints_frame, text=f"V (km/s): {v_min:.0f} to {v_max:.0f}").grid(
            row=0, column=0, sticky="w", padx=10, pady=4
        )
        ttk.Label(constraints_frame, text=f"By (nT): {by_min:.0f} to {by_max:.0f}").grid(
            row=0, column=1, sticky="w", padx=10, pady=4
        )
        ttk.Label(constraints_frame, text=f"Bz (nT): {bz_min:.0f} to {bz_max:.0f}").grid(
            row=0, column=2, sticky="w", padx=10, pady=4
        )

        ttk.Label(self, text="Solar wind speed V (km/s):").grid(
            row=2, column=0, sticky="w", **pad
        )
        ttk.Entry(self, textvariable=self.v_var, width=20).grid(
            row=2, column=1, sticky="w", **pad
        )

        ttk.Label(self, text="IMF By (nT):").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.by_var, width=20).grid(
            row=3, column=1, sticky="w", **pad
        )

        ttk.Label(self, text="IMF Bz (nT):").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.bz_var, width=20).grid(
            row=4, column=1, sticky="w", **pad
        )

        ttk.Button(self, text="Compute", command=self._on_compute).grid(
            row=5, column=0, sticky="w", **pad
        )
        ttk.Button(self, text="Reset", command=self._on_reset).grid(
            row=5, column=1, sticky="w", **pad
        )

        ttk.Separator(self, orient="horizontal").grid(
            row=6, column=0, columnspan=3, sticky="ew", padx=10, pady=10
        )

        ttk.Label(self, text="Derived Bt (nT):").grid(row=7, column=0, sticky="w", **pad)
        ttk.Label(self, textvariable=self.bt_var).grid(row=7, column=1, sticky="w", **pad)

        ttk.Label(self, text="Clock angle theta (deg):").grid(
            row=8, column=0, sticky="w", **pad
        )
        ttk.Label(self, textvariable=self.theta_var).grid(
            row=8, column=1, sticky="w", **pad
        )

        ttk.Label(self, text="Newell coupling dPhi/dt (proxy):").grid(
            row=9, column=0, sticky="w", **pad
        )
        ttk.Label(self, textvariable=self.coupling_var, font=("Segoe UI", 10, "bold")).grid(
            row=9, column=1, sticky="w", **pad
        )

        ttk.Label(self, textvariable=self.status_var, foreground="#1b5e20").grid(
            row=10, column=0, columnspan=3, sticky="w", **pad
        )

        ttk.Label(self, text="Coupling Rate Curve vs Bz (current V and By)", font=("Segoe UI", 10, "bold")).grid(
            row=11, column=0, columnspan=3, sticky="w", padx=10, pady=(6, 2)
        )

        self.plot_canvas = tk.Canvas(
            self,
            width=self.plot_width,
            height=self.plot_height,
            bg="white",
            highlightthickness=1,
            highlightbackground="#aaaaaa",
        )
        self.plot_canvas.grid(row=12, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 10))

    def _draw_graph(self, v_kms, by_nt, bz_nt, coupling_now):
        """Draw coupling-rate curve as Bz sweeps across acceptable range."""
        if self.plot_canvas is None:
            return

        self.plot_canvas.delete("all")

        x0 = self.plot_padding
        y0 = self.plot_height - self.plot_padding
        x1 = self.plot_width - self.plot_padding
        y1 = self.plot_padding

        bz_min, bz_max = ACCEPTABLE_RANGES["bz_nt"]
        bz_grid = np.linspace(bz_min, bz_max, 240)
        coupling_curve, _, _ = newells_coupling_function(v_kms, by_nt, bz_grid)

        y_min = 0.0
        y_max = float(np.max(coupling_curve)) if np.max(coupling_curve) > 0 else 1.0

        def to_canvas_x(xval):
            return x0 + (xval - bz_min) / (bz_max - bz_min) * (x1 - x0)

        def to_canvas_y(yval):
            return y0 - (yval - y_min) / (y_max - y_min) * (y0 - y1)

        # Axes
        self.plot_canvas.create_line(x0, y0, x1, y0, fill="#222222", width=2)
        self.plot_canvas.create_line(x0, y0, x0, y1, fill="#222222", width=2)

        # Axis labels and limits
        self.plot_canvas.create_text((x0 + x1) / 2, y0 + 24, text="Bz (nT)", fill="#333333")
        self.plot_canvas.create_text(x0 - 28, (y0 + y1) / 2, text="dPhi/dt", fill="#333333", angle=90)
        self.plot_canvas.create_text(x0, y0 + 12, text=f"{bz_min:.0f}", anchor="nw", fill="#444444")
        self.plot_canvas.create_text(x1, y0 + 12, text=f"{bz_max:.0f}", anchor="ne", fill="#444444")
        self.plot_canvas.create_text(x0 - 6, y0, text=f"{y_min:.1f}", anchor="e", fill="#444444")
        self.plot_canvas.create_text(x0 - 6, y1, text=f"{y_max:.1f}", anchor="e", fill="#444444")

        # Coupling curve polyline
        points = []
        for bz_val, c_val in zip(bz_grid, coupling_curve):
            points.extend([to_canvas_x(float(bz_val)), to_canvas_y(float(c_val))])
        self.plot_canvas.create_line(*points, fill="#0b6efd", width=2, smooth=True)

        # Highlight current input point on the curve.
        px = to_canvas_x(bz_nt)
        py = to_canvas_y(float(coupling_now))
        self.plot_canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#d63384", outline="#d63384")
        self.plot_canvas.create_text(
            px + 8,
            py - 8,
            text=f"Current: Bz={bz_nt:.2f}, dPhi/dt={float(coupling_now):.2f}",
            anchor="w",
            fill="#d63384",
        )

    def _on_compute(self):
        """Read inputs, validate range limits, run NumPy coupling calculation, update UI."""
        try:
            v_kms = parse_input_with_range("v_kms", self.v_var.get())
            by_nt = parse_input_with_range("by_nt", self.by_var.get())
            bz_nt = parse_input_with_range("bz_nt", self.bz_var.get())

            coupling, bt_nt, theta_deg = newells_coupling_function(v_kms, by_nt, bz_nt)

            self.bt_var.set(f"{float(bt_nt):.3f}")
            self.theta_var.set(f"{float(theta_deg):.3f}")
            self.coupling_var.set(f"{float(coupling):.3f}")
            self.status_var.set("Computation successful.")
            self._draw_graph(v_kms, by_nt, bz_nt, coupling)
        except ValueError as err:
            self.status_var.set(str(err))

    def _on_reset(self):
        """Restore defaults and clear output fields."""
        self.v_var.set("450")
        self.by_var.set("5")
        self.bz_var.set("-8")
        self.bt_var.set("-")
        self.theta_var.set("-")
        self.coupling_var.set("-")
        self.status_var.set("Enter values and click Compute.")
        self._on_compute()

if __name__ == "__main__":
    app = NewellCouplingGUI()
    app.mainloop()
