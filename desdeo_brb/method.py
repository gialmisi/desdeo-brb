import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.backends.backend_tkagg as tkagg
import pandas
from pandas.plotting import parallel_coordinates

import matplotlib
matplotlib.use("tkagg")

import matplotlib.pyplot as plt

from brb import BRBPref
from utility import (load_and_scale_data,
                     plot_utility_monotonicity,
                     plot_3d_ranks_colored,
                     simple_mapping,
                     brb_score)

const_font = "Arial Bold"
const_font_size = 12
const_fontsetting = (const_font, const_font_size)
variant = 1

objective_names = ["Income", "Stored CO2", "CHSI"]


def method(data_dir: str, fname_po: str, fname_pf: str, objective_names: str = objective_names):
    # setup tkinter app instance
    window = tk.Tk()
    window.title(f"INFRINGER - Variant {variant}")

    # load pre-computed paretofront, nadir, ideal, and the scaler
    # the data is scaled between 0 and 1 using the nadir and ideal points
    nadir, ideal, paretofront, payoff, scaler = load_and_scale_data(data_dir, fname_po, fname_pf)
    nadir = np.squeeze(nadir)
    ideal = np.squeeze(ideal)
    # used to store a reference point given by the DM
    global reference_point
    reference_point = None

    # show ideal and nadir
    def make_table(row_names: [str], col_names: [str], data: np.ndarray, row=1, column=0):
        tframe = tk.Frame(window, height=700, width=500, borderwidth=2)
        tframe.grid(row=row, column=column)

        entry = tk.Label(tframe, text="", font=const_fontsetting)
        entry.grid(row=0, sticky="we")
        
        # set row names
        for i, name in enumerate(row_names):
            entry = tk.Label(tframe, text=f"{name}", font=const_fontsetting)
            entry.grid(row=i+1, column=0, padx=20, sticky="we")

        # set column names
        for i, name in enumerate(col_names):
            entry = tk.Label(tframe, text=f"{name}", font=const_fontsetting)
            entry.grid(row=0, column=i+1, sticky="we")

        # set contents
        for i, _ in enumerate(row_names):
            for j, _ in enumerate(col_names):
                print(f"{(i, j)}")
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
    make_table(["nadir", "ideal"], objective_names, scaler.inverse_transform(np.stack((nadir, ideal))))

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
        reference_point = scaler.transform(reference_point.reshape(1, -1)).squeeze()
        print(f"ref point normed: {reference_point}")
    else:
        print("ref point not given")
        reference_point = np.array([0.5, 0.5, 0.5])
        print(f"ref point is {reference_point}")

    # show the paretofront
    fig_pf = plt.figure(figsize=(8, 6), dpi=160)
    fig_pf.suptitle("Paretofront in original scale")

    canvas_pf = tkagg.FigureCanvasTkAgg(fig_pf, master=window)

    ax_pf = fig_pf.add_subplot(111, projection="3d")
    extrema = scaler.inverse_transform(np.stack((nadir, ideal)))
    ax_pf.scatter(extrema[0, 0], extrema[0, 1], extrema[0, 2], label="nadir")
    ax_pf.scatter(extrema[1, 0], extrema[1, 1], extrema[1, 2], label="ideal")
    reference_point_orig = scaler.inverse_transform(reference_point.reshape(1, -1)).squeeze()
    ax_pf.scatter(reference_point_orig[0],
                  reference_point_orig[1],
                  reference_point_orig[2],
                  label="reference point")
    paretofront_orig = scaler.inverse_transform(paretofront)
    ax_pf.scatter(paretofront_orig[:, 0], paretofront_orig[:, 1], paretofront_orig[:, 2])
    ax_pf.set_xlabel("Income")
    ax_pf.set_ylabel("Carbon")
    ax_pf.set_zlabel("CHSI")
    fig_pf.legend()

    canvas_pf.draw()
    canvas_pf.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar_pf = tkagg.NavigationToolbar2Tk(canvas_pf, window)
    toolbar_pf.update()
    canvas_pf.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    btn_next = tk.Button(window, text="Next", command=lambda: quit(window))
    btn_next.pack(side=tk.BOTTOM, fill=tk.X)

    # show the 3D plot
    window.mainloop()

    # clear main window
    [child.destroy() for child in window.winfo_children()]

    # initialize BRB
    # define parameters for the BRB
    precedents = np.stack((nadir, reference_point, ideal)).T
    consequents = np.array([[0, 0.25, 0.5, 0.75, 1]])

    brb = BRBPref(precedents, consequents, f=lambda x: simple_mapping(x))

    if variant == 1:
        pf_scores = brb_score(brb, paretofront)

        pf_scores_mean = np.mean(pf_scores)
        pf_scores_std = np.std(pf_scores)

        # show pairs to compare and asses fitness  of model
        dm_choices = ask_preference(window, brb, paretofront, scaler, nadir=nadir, ideal=ideal)

        # check compatibility
        fitness, brb_choices = calculate_fitness(window, brb, dm_choices) 

        # clear main window
        [child.destroy() for child in window.winfo_children()]

        # find 5 evenly distributed points (score wise) around the mean to be scored by the DM
        target_scores = np.array([pf_scores_mean - pf_scores_std,
                                pf_scores_mean - pf_scores_std/2,
                                pf_scores_mean,
                                pf_scores_mean + pf_scores_std/2,
                                pf_scores_mean + pf_scores_std])

        diff_to_target = np.abs(pf_scores[:, None] - target_scores)
        score_indices = np.argmin(diff_to_target, axis=0)
        points_to_compare = paretofront[score_indices]

        # ask DM to score the points
        lbl_title = tk.Label(window, text="Please score each candidate between 0(worst) and 100(best)", font=const_fontsetting)
        lbl_title.grid(row=0, columnspan=3)
        lbl_scoring = tk.Label(window,
                            text="Candidates",
                            font=const_fontsetting)
        lbl_scoring.grid(row=1, column=0, sticky="we")


        make_table([f"Candidate {i+1}" for i in range(len(points_to_compare))],
                objective_names,
                scaler.inverse_transform(points_to_compare),
                row=1, column=0)

        lbl_score = tk.Label(window, text="Score", font=const_fontsetting)
        lbl_score.grid(row=1, column=1, sticky="we")

        frame_scoring = tk.Frame(window)
        frame_scoring.grid(row=1, column=1)
        lbl_scoring = tk.Label(frame_scoring, text="Scores", font=const_fontsetting)
        lbl_scoring.grid(row=0)

        candidate_scores_entries = []
        for i in range(len(points_to_compare)):
            entry_score = tk.Entry(frame_scoring, font=const_fontsetting)
            entry_score.grid(row=1+i, column=0, sticky="we", pady=2.5)
            entry_score.insert(tk.END, f"{np.random.randint(0, 101)}")
            candidate_scores_entries.append(entry_score)

        # show nadir and ideal
        lbl_nadirandideal = tk.Label(window,
                            text="Nadir and ideal points",
                            font=const_fontsetting)
        lbl_nadirandideal.grid(row=1, column=2, sticky="we")

        make_table(["nadir", "ideal"], objective_names, scaler.inverse_transform(np.stack((nadir, ideal))), row=1, column=2)

        no_btn = tk.Button(window, text="Cancel", command=close_window)
        no_btn.grid(row=4, column=2)

        btn_plot = tk.Button(window, text="Plot", command=lambda: draw_parallel(points_to_compare, obj_names=objective_names))
        btn_plot.grid(row=4, column=4)

        global candidate_scores
        candidate_scores = None
        def check_scores():
            entry_txt_list = [entry.get() for entry in candidate_scores_entries]
            try:
                res = [int(txt) for txt in entry_txt_list]
                res = np.array(res)/100
            except ValueError as e:
                tk.messagebox.showerror("Error in parsing candidate scores",
                                f"Could not parse {entry_txt_list}. Are you sure the entries are integers?")
                return False

            if not np.all(0 <= res) or not np.all(res <= 1):
                tk.messagebox.showerror("Invalid candidate score(s)",
                                        "The given candidate score(s)is not between the 0 and 100")
                return False

            # valid scores given
            global candidate_scores
            candidate_scores = res
            window.quit()

            return True

        next_btn = tk.Button(window, text="Next", command=check_scores)
        next_btn.grid(row=4, column=1)

        window.mainloop()

        # clear main window
        [child.destroy() for child in window.winfo_children()]

        tk.messagebox.showinfo("Training...", "The BRB model will be trained now, this might take a while.")

        # train BRB
        brb.train(None,
                None,
                brb._flatten_parameters(),
                obj_args=(np.atleast_2d(nadir),
                            np.atleast_2d(ideal),
                            np.atleast_2d(points_to_compare),
                            np.atleast_2d(candidate_scores)),
                  use_de=True,
        )

        print(brb)
        print(brb_score(brb, np.atleast_2d(points_to_compare)))
        print(candidate_scores)

    if variant == 2:
        # check fitness again
        dm_choices = ask_preference(window, brb, paretofront, scaler, nadir=nadir, ideal=ideal)
        fitness, brb_choices = calculate_fitness(window, brb, dm_choices)

        if fitness >= 80:
            tk.messagebox.showinfo("Success", "Success! Fitness of over 80% achieveved!")
            print("Top 5 best solutions are:")
            pf_scores = brb_score(brb, paretofront)
            best_5 = np.argsort(pf_scores)[::-1][:5]
            print(f"{paretofront_orig[best_5]}\n with scores {pf_scores[best_5]}")
            return

        tk.messagebox.showinfo("Training...", "The BRB model will be trained now, this might take a while.")

        brb.train(None,
                None,
                brb._flatten_parameters(),
                obj_args=(np.atleast_2d(nadir),
                            np.atleast_2d(ideal),
                            None,
                            None,
                            dm_choices),
                  use_de=True,
        )

    while True:
        # clear main window
        [child.destroy() for child in window.winfo_children()]

        # show value funtion
        draw_utility(window, brb)
        window.mainloop()

        # clear main window
        [child.destroy() for child in window.winfo_children()]

        # show ranking
        draw_ranking(window, brb, paretofront)
        window.mainloop()

        # clear main window
        [child.destroy() for child in window.winfo_children()]

        # check fitness again
        dm_choices = ask_preference(window, brb, paretofront, scaler, nadir=nadir, deal=ideal)
        fitness, brb_choices = calculate_fitness(window, brb, dm_choices)

        if fitness >= 80:
            tk.messagebox.showinfo("Success", "Success! Fitness of over 80% achieveved!")
            print("Top 5 best solutions are:")
            pf_scores = brb_score(brb, paretofront)
            best_5 = np.argsort(pf_scores)[::-1][:5]
            print(f"{paretofront_orig[best_5]}\n with scores {pf_scores[best_5]}")
            break

        tk.messagebox.showinfo("Training...", "The BRB model will be trained now, this might take a while.")

        if variant == 1:
            brb.train(None,
                    None,
                    brb._flatten_parameters(),
                    obj_args=(np.atleast_2d(nadir),
                                np.atleast_2d(ideal),
                                np.atleast_2d(points_to_compare),
                                np.atleast_2d(candidate_scores),
                                dm_choices),
                      use_de=True,
            )
        if variant == 2:
            brb.train(None,
                    None,
                    brb._flatten_parameters(),
                    obj_args=(np.atleast_2d(nadir),
                                np.atleast_2d(ideal),
                                None,
                                None,
                                dm_choices),
                      use_de=True,
            )

        print(brb)

    return 0


def ask_preference(window, brb, paretofront, scaler, objective_names=["Income", "Stored CO2", "CHSI"], nadir=None, ideal=None):
    frame_comp = tk.Frame(window)
    frame_comp.grid(row=0)

    lbl_title = tk.Label(frame_comp, text="Please specify your preference",
                         font=const_fontsetting)
    lbl_title.grid(row=0, columnspan=7)

    btn_quit = tk.Button(frame_comp, text="Quit", command=lambda: quit(window))
    btn_quit.grid(row=8, column=1)


    pairs = []
    brb_scores = brb_score(brb, paretofront)

    mean = np.mean(brb_scores)
    std = np.std(brb_scores)

    target_scores_fst = np.array([mean + std/4,
                                  mean + std/3,
                                  mean + std/2,
                                  1,
                                  np.random.uniform()])
    target_scores_snd = np.array([mean - std/4,
                                  mean - std/3,
                                  mean - std/2,
                                  0,
                                  np.random.uniform()])

    target_candidates_fst_ = paretofront[np.argmin(np.abs(brb_scores[:, None] - target_scores_fst), axis=0)]
    target_candidates_snd_ = paretofront[np.argmin(np.abs(brb_scores[:, None] - target_scores_snd), axis=0)]
    target_candidates_fst = scaler.inverse_transform(target_candidates_fst_)
    target_candidates_snd = scaler.inverse_transform(target_candidates_snd_)

    cbs = []
    for i in range(len(target_candidates_fst)):
        lbl_name = tk.Label(frame_comp, text=f"Candidates {i+1}")
        lbl_name.grid(row=i+2, column=0)
        # each row
        if i == 0:
            for k, name in enumerate(objective_names):
                lbl_name = tk.Label(frame_comp, text=name)
                lbl_name.grid(row=1, column=k+1)

                lbl_name = tk.Label(frame_comp, text=name)
                lbl_name.grid(row=1, column=k+2+len(target_candidates_fst[0]))

        for j in range(len(target_candidates_fst[0])):
            # each column
            lbl = tk.Label(frame_comp,
                           text=f"{np.format_float_scientific(target_candidates_fst[i, j], precision=5)}",
                           font=const_fontsetting, relief="ridge")
            lbl.grid(row=2+i, column=j+1, sticky="we", padx=5, pady=2.5)

        cb_choice = ttk.Combobox(frame_comp, font=const_fontsetting, state="readonly", justify=tk.CENTER)
        cb_choice["values"] = ("is better than", "is worse than", "is as good as")
        cb_choice.current(2)
        cb_choice.grid(row=2+i, column=1+len(target_candidates_fst[0]),
                       sticky="we", padx=5, pady=2.5)
        cbs.append(cb_choice)

        for j in range(len(target_candidates_snd[0])):
            # each column
            lbl = tk.Label(frame_comp,
                           text=f"{np.format_float_scientific(target_candidates_snd[i, j], precision=5)}",
                           font=const_fontsetting, relief="ridge")
            lbl.grid(row=2+i, column=j+2+len(target_candidates_snd[0]), sticky="we", padx=5, pady=2.5)

        btn_show = tk.Button(frame_comp, text="Plot",
                             command=lambda i=i:
                             draw_radial(np.stack((target_candidates_fst_[i], target_candidates_snd_[i])),
                             name=f"Candidates {i+1}",
                             nadir=nadir,
                             ideal=ideal,
                             
        ))
        btn_show.grid(row=2+i, column=2*len(target_candidates_fst[0])+3, sticky="we", padx=5, pady=2.5)

    global choices
    choices = []
    def get_choices(cbs):
        global choices
        choices_txt = [cb.get() for cb in cbs]
        for i, txt in enumerate(choices_txt):
            if txt == "is better than":
                pref = 1
            elif txt == "is worse than":
                pref = 2
            else:
                pref = 0

            choices.append((target_candidates_fst_[i],
                           target_candidates_snd_[i],
                           pref))
        quit(window)
        return True
                

    btn_next = tk.Button(frame_comp, text="Next", command=lambda: get_choices(cbs))
    btn_next.grid(row=8, column=len(objective_names)+1, sticky="we")

    window.mainloop()

    # return a list of tuples with the first candidate of a pair, the second candiate, and preference:
    # preference: 0 as good as, 1 first is better, 2 second is better
    return choices


def calculate_fitness(window, brb, dm_choices, delta=0.05):
    brb_choices = []
    fitness = 0
    for choice in dm_choices:
        brb_score_fst = brb_score(brb, np.atleast_2d(choice[0]))[0]
        brb_score_snd = brb_score(brb, np.atleast_2d(choice[1]))[0]

        if brb_score_fst > brb_score_snd:
            brb_pref = 1
        elif brb_score_fst < brb_score_snd:
            brb_pref = 2
        elif abs(brb_score_fst - brb_score_snd) < delta:
            brb_pref = 0

        brb_choices.append((
            choice[0],
            choice[1],
            brb_pref
        ))

        if brb_pref == choice[-1]:
            fitness += 1

    # scale fitness between 0 and 100
    fitness /= len(dm_choices)
    fitness *= 100

    d = {1: "first", 2: "second", 0: "indifferent"}
    for i in range(len(brb_choices)):
        print(f"For pair n. {i+1}: BRB preference: {d[brb_choices[i][-1]]} "
              f"\tand DM preference: {d[dm_choices[i][-1]]}")

    tk.messagebox.showinfo("Fitness", f"The BRB model agrees with you {int(fitness)}%!")

    return fitness, brb_choices



    
def draw_ranking(window, brb, paretofront):
    fig_3d = plt.figure(figsize=(8, 6), dpi=160)
    canvas_res = tkagg.FigureCanvasTkAgg(fig_3d, master=window)
    plot_3d_ranks_colored(brb, paretofront, fig=fig_3d)

    canvas_res.draw()
    canvas_res.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar_res = tkagg.NavigationToolbar2Tk(canvas_res, window)
    toolbar_res.update()
    canvas_res.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    btn_next = tk.Button(window, text="Next", command=lambda: quit(window))
    btn_next.pack(side=tk.BOTTOM, fill=tk.X)

    return True


def draw_utility(window, brb):
    fig_util = plt.figure(figsize=(9, 6), dpi=160)
    canvas_res = tkagg.FigureCanvasTkAgg(fig_util, master=window)
    plot_utility_monotonicity(brb, [[0,1], [0,1], [0,1]], fig=fig_util)

    canvas_res.draw()
    canvas_res.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar_res = tkagg.NavigationToolbar2Tk(canvas_res, window)
    toolbar_res.update()
    canvas_res.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    btn_next = tk.Button(window, text="Next", command=lambda: quit(window))
    btn_next.pack(side=tk.BOTTOM, fill=tk.X)

    return True


def draw_radial(candidates, name=None, nadir=None, ideal=None):
    window = tk.Tk()
    window.title(f"Candidate comparison")

    fig_radial = plt.figure(figsize=(5, 4), dpi=160)
    canvas_res = tkagg.FigureCanvasTkAgg(fig_radial, master=window)

    if name is not None:
        fig_radial.suptitle(name)

    data = np.zeros((candidates.shape[0], candidates.shape[1]+1))
    data[:, :-1] = candidates
    data[:, -1] = candidates[:, 0]

    n_vars = candidates.shape[1]

    angles = [n/n_vars*2*np.pi for n in range(n_vars)]
    angles += angles[:1]

    ax = fig_radial.add_subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], objective_names)

    ax.set_rlabel_position(0)
    plt.yticks([0, 1], ["nadir", "ideal"], color="grey", size=7)
    plt.ylim(-.25, 1)

    ax.plot(angles, data[0], linewidth=1, linestyle="solid", label="Candidate 1")
    ax.fill(angles, data[0], "b", alpha=0.1)
    ax.plot(angles, data[1], linewidth=1, linestyle="solid", label="Candidate 2")
    ax.fill(angles, data[1], "r", alpha=0.1)

    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    fig_radial.tight_layout(rect=[0, 0.03, 1, 0.95])

    canvas_res.draw()
    canvas_res.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar_res = tkagg.NavigationToolbar2Tk(canvas_res, window)
    toolbar_res.update()
    canvas_res.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    btn_close = tk.Button(window, text="Close", command=lambda: window.destroy())
    btn_close.pack(side=tk.BOTTOM, fill=tk.X)

    return True


def draw_parallel(candidates, obj_names=["INCOME", "CO2", "CHSI"]):
    window = tk.Tk()
    window.title(f"Candidate comparison")

    fig_par = plt.figure(figsize=(7, 4), dpi=160)
    canvas_res = tkagg.FigureCanvasTkAgg(fig_par, master=window)
    
    fig_par.suptitle("Candidate comparison")
    ax = fig_par.add_subplot(111)
    
    df = pandas.DataFrame(data=candidates,
                        index=[f"Candidate {i+1}" for i in range(len(candidates))],
                        columns=obj_names).reset_index()


    if len(candidates) == 5:
        parallel_coordinates(df, "index", ax=ax, colors=["red", "green", "grey", "blue", "orange"])
    else:
        parallel_coordinates(df, "index", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["nadir", "", "", "", "ideal"])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.65, box.height])
    legend_x = 1
    legend_y = 0.5
    ax.legend(loc="center left", bbox_to_anchor=(legend_x, legend_y))

    canvas_res.draw()
    canvas_res.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar_res = tkagg.NavigationToolbar2Tk(canvas_res, window)
    toolbar_res.update()
    canvas_res.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    btn_close = tk.Button(window, text="Close", command=lambda: window.destroy())
    btn_close.pack(side=tk.BOTTOM, fill=tk.X)

    return True


def quit(window):
    window.quit()


if __name__=="__main__":
    #candidates = np.array([[0.5, 0.75, 0.25],
    #                       [0.22, 0.55, 0.82]])
    #draw_radial(candidates)
    method("/home/kilo/workspace/forest-opt/data/",
            "payoff.dat",
            "test_run.dat")
