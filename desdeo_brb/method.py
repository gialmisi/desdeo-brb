import numpy as np
import tkinter as tk
from brb import BRBPref
from utility import load_and_scale_data, plot_utility_monotonicity, plot_3d_ranks_colored

const_font = "Arial Bold"
const_font_size = 12
const_fontsetting = (const_font, const_font_size)


def method1(data_dir: str, fname_po: str, fname_pf: str, objective_names: str = ["Income", "Stored CO2", "CHSI"]):
    # setup tkinter app instance
    window = tk.Tk()
    window.title("INFRINGER - Variant 1")

    # load pre-computed paretofront, nadir, ideal, and the scaler
    # the data is scaled between 0 and 1 using the nadir and ideal points
    nadir, ideal, paretofront, payoff, scaler = load_and_scale_data(data_dir, fname_po, fname_pf)
    nadir = np.squeeze(nadir)
    ideal = np.squeeze(ideal)
    # used to store a reference point given by the DM
    global reference_point
    reference_point = None

    # show ideal and nadir
    def make_table(row_names: str, col_names: str, data: np.ndarray):
        tframe = tk.Frame(window, height=700, width=500, borderwidth=2)
        tframe.grid(row=1)

        entry = tk.Label(tframe, text="", font=const_fontsetting)
        entry.grid(row=0, sticky="we")
        
        # set row names
        for i, name in enumerate(row_names):
            entry = tk.Label(tframe, text=f"{name}", font=const_fontsetting)
            entry.grid(row=0, column=i+1, padx=20, sticky="we")

        # set column names
        for i, name in enumerate(col_names):
            entry = tk.Label(tframe, text=f"{name}", font=const_fontsetting)
            entry.grid(row=i+1, column=0, sticky="we")

        # set contents
        for i, _ in enumerate(col_names):
            for j, _ in enumerate(row_names):
                entry = tk.Label(tframe, text=f"{np.format_float_scientific(data[i, j], precision=5)}",
                                 font=const_fontsetting, relief="ridge")
                entry.grid(row=i+1, column=j+1, sticky="we", padx=5, pady=2.5)
            pass

        return True

    def close_window(frame=window):
        frame.quit()
        return True

    def give_reference_point(nadir: np.ndarray, ideal: np.ndarray, to_destroy=None):
        # destroy frames
        [thing.destroy() for thing in to_destroy]

        # build a reference form
        reference_frame = tk.Frame(window)
        reference_frame.grid(row=4)

        lbl = tk.Label(reference_frame, text="Please specify a reference point with objective values between the nadir and ideal.",
                       font=const_fontsetting)
        lbl.grid(row=0, column=0, columnspan=len(nadir))

        reference_point_entry_list = []

        for i in range(len(nadir)):
            print(i)
            lbl_ref = tk.Label(reference_frame, text=f"Objective {i+1}:")
            lbl_ref.grid(row=1, column=i)

            txt_entry = tk.Entry(reference_frame, font=const_fontsetting)
            txt_entry.insert(tk.END, f"{nadir[i]}")
            txt_entry.grid(row=2, column=i, sticky="we")
            reference_point_entry_list.append(txt_entry)

        def check_reference_point():
            entry_txt_list = [entry.get() for entry in reference_point_entry_list]
            try:
                res = [float(txt) for txt in entry_txt_list]
                res = np.array(res)
            except ValueError as e:
                tk.messagebox.showerror("Error in parsing reference point",
                              f"Could not parse {entry_txt_list}. Are you sure the entries are numerical?")
                return False

            if not np.all(nadir <= res) or not np.all(res <= ideal):
                tk.messagebox.showerror("Invalid reference point",
                                        "The given reference point is not between the nadir and ideal points")
                return False

            # good reference point given
            global reference_point
            reference_point = res
            window.quit()

            return True

        btn_check = tk.Button(reference_frame, text="Check", command=check_reference_point)
        btn_check.grid(row=3, columnspan=len(nadir), sticky="we", pady=5)

        return True


    # Start drawing on main window
    lbl_idealandnadir = tk.Label(window, text=f"Ideal and nadir points", font=const_fontsetting)
    lbl_idealandnadir.grid(row=0, padx=250)

    # show ideal and nadir
    make_table(objective_names, ["nadir", "ideal"], scaler.inverse_transform(np.stack((nadir, ideal))))

    # ask if DM wants to give a ref point
    lbl_question = tk.Label(window, text="Would you like to give a reference point?", pady=5, font=const_fontsetting)
    lbl_question.grid(row=2)

    btn_frame = tk.Frame(window)
    btn_frame.grid(row=3)

    yes_btn = tk.Button(btn_frame, text="Yes",
                        command=lambda: give_reference_point(scaler.inverse_transform(nadir.reshape(1, -1)).squeeze(),
                                                             scaler.inverse_transform(ideal.reshape(1, -1)).squeeze(),
                                                             [btn_frame, lbl_question]))
    yes_btn.grid(row=0, column=1)

    no_btn = tk.Button(btn_frame, text="No", command=close_window)
    no_btn.grid(row=0, column=0)

    # show window until DM gives valid ref point or chooses not to
    window.mainloop()

    # clear the main window
    [child.destroy() for child in window.winfo_children()]

    # normalize reference point if given
    if reference_point is not None:
        print(f"ref point given: {reference_point}")
        reference_point = scaler.transform(reference_point.reshape(1, -1))
        print(f"ref point normed: {reference_point}")
    else:
        print("ref point not given")
        reference_point = np.array([0.5, 0.5, 0.5])
        print(f"ref point is {reference_point}")

    # show the paretofront

    window.mainloop()


if __name__=="__main__":
    method1("/home/kilo/workspace/forest-opt/data/",
            "payoff.dat",
            "test_run.dat")
