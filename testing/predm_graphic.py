import matplotlib.pyplot as plt
import numpy as np


def text_box(ax, string, pos_x):
    # these are matplotlib.patch.Patch properties
    props = {'facecolor': 'white', 'alpha': 1, 'edgecolor': 'none', 'pad': 1}
    # props = {'boxstyle': 'round', 'facecolor': 'lightblue', 'alpha': 0.75, 'edgecolor': 'none', 'pad': 1,}

    ax.text(pos_x, 0.5, string, bbox=props, ha='center', va='center', fontsize=12)


def main():
    x_range = 10
    x = np.linspace(0.1, x_range)

    cost_repair = lambda x: np.power(np.e, -0.2 * (x - 5))
    cost_prev = lambda x: np.power(np.e, 0.2 * (x - 5))
    cost_total = lambda x: (cost_repair(x) + cost_prev(x))

    rc = {"xtick.direction": "inout", "ytick.direction": "inout",
          "xtick.major.size": 5, "ytick.major.size": 5, }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.axvline(x=x_range / 3, color='k', linestyle='--', ymin=0.075, ymax=0.975, dashes=(3, 4))
        ax.axvline(x=x_range / 3 * 2, color='k', linestyle='--', ymin=0.075, ymax=0.975, dashes=(3, 4))

        # ax.set_xlim([0, 10])
        ax.axis([None, None, -0.3, 4.5])

        c = 0.75
        ax.plot(x, cost_total(x) + c, label='Gesamtkosten')
        ax.plot(x, cost_prev(x) + c, label='Wartungskosten')
        ax.plot(x, cost_repair(x) + c, label='Reparaturkosten')
        ax.set_xlabel('Frequenz der Wartungsarbeiten', labelpad=10, fontsize=12)
        ax.set_ylabel('Kosten', labelpad=10,fontsize=12)

        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')

        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        ax.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False,
            labelright=False)  # labels along the bottom edge are off

        text_box(ax, 'Ausfallbasierte oder\nseltene präventive\nInstandhaltung', 10 / 6)
        text_box(ax, 'Zustandsorientierte\nInstandhaltung', 10 / 2)
        text_box(ax, 'Häufige präventive\nInstandhaltung', 10 / 6 * 5)

        # make arrows
        ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                transform=ax.get_xaxis_transform(), clip_on=False)

        fig.tight_layout()
        plt.legend(loc='upper center', fontsize=12)
        plt.savefig('predm.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    main()
