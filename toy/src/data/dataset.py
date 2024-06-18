import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import sklearn.datasets as datasets

# Mixture of Uniform to mixture of gaussians


class mixture_uni_to_gaussians_v2(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf.
    """

    def __init__(self, n_samples=100):
        super(mixture_uni_to_gaussians_v2, self).__init__()
        self.n_samples = n_samples
        self.sigma1 = 0.1
        self.sigma2 = 0.25
        self.sigma3 = 0.05
        self.sigma4 = 0.1

        self.s1 = (-1, 0, -1, 0)
        self.s2 = (-1, 0, 0, 1)
        self.s3 = (0, 1, -1, 0)
        self.s4 = (0, 1, 0, 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)
        # s1 = (-1, 0, -1, 0)
        s1 = self.s1
        s2 = self.s2
        s3 = self.s3
        s4 = self.s4

        # Select a section according to the probabilities defined in the paper
        # section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

        choice_distribution = np.random.choice([1, 2], p=[(1 - t), t])

        if choice_distribution == 1:

            p1 = p4 = 1 / 2
            p2 = p3 = 0

            section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

            # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
            # Sample a point uniformly from the selected section
            if section == 1:
                x = np.random.uniform(s1[0], s1[1])
                y = np.random.uniform(s1[2], s1[3])
            elif section == 2:
                x = np.random.uniform(s2[0], s2[1])
                y = np.random.uniform(s2[2], s2[3])
            elif section == 3:
                x = np.random.uniform(s3[0], s3[1])
                y = np.random.uniform(s3[2], s3[3])
            elif section == 4:
                x = np.random.uniform(s4[0], s4[1])
                y = np.random.uniform(s4[2], s4[3])
            else:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
            #################

        elif choice_distribution == 2:

            p1 = p4 = 0
            p2 = p3 = 1 / 2

            section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

            # origin = np.array([0,0])
            mean1 = np.array([-1 / 2, -1 / 2])
            mean2 = np.array([-1 / 2, 1 / 2])
            mean3 = np.array([1 / 2, -1 / 2])
            mean4 = np.array([1 / 2, 1 / 2])

            # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

            # Sample a point uniformly from the selected section
            if section == 1:
                x, y = np.random.multivariate_normal(mean1, np.eye(2) * self.sigma1**2)
            elif section == 2:
                x, y = np.random.multivariate_normal(mean2, np.eye(2) * self.sigma2**2)
            elif section == 3:
                x, y = np.random.multivariate_normal(mean3, np.eye(2) * self.sigma3**2)
            elif section == 4:
                x, y = np.random.multivariate_normal(mean4, np.eye(2) * self.sigma4**2)

        return torch.Tensor([t, x, y])

    def generate_dataset_distribution(self, t, n_samples, plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        # s1 = (-1, 0, -1, 0)
        s1 = self.s1
        s2 = self.s2
        s3 = self.s3
        s4 = self.s4

        # Define the probabilities of selecting each section
        # p1 = p4 = (1 - t) / 2
        # p2 = p3 = t / 2

        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            choice_distribution = np.random.choice([1, 2], p=[(1 - t), t])

            if choice_distribution == 1:

                p1 = p4 = 1 / 2
                p2 = p3 = 0

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
                # Sample a point uniformly from the selected section
                if section == 1:
                    x = np.random.uniform(s1[0], s1[1])
                    y = np.random.uniform(s1[2], s1[3])
                elif section == 2:
                    x = np.random.uniform(s2[0], s2[1])
                    y = np.random.uniform(s2[2], s2[3])
                elif section == 3:
                    x = np.random.uniform(s3[0], s3[1])
                    y = np.random.uniform(s3[2], s3[3])
                elif section == 4:
                    x = np.random.uniform(s4[0], s4[1])
                    y = np.random.uniform(s4[2], s4[3])
                else:
                    x = np.random.uniform(-1, 1)
                    y = np.random.uniform(-1, 1)
                #################

            elif choice_distribution == 2:

                p1 = p4 = 0
                p2 = p3 = 1 / 2

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                # origin = np.array([0,0])
                mean1 = np.array([-1 / 2, -1 / 2])
                mean2 = np.array([-1 / 2, 1 / 2])
                mean3 = np.array([1 / 2, -1 / 2])
                mean4 = np.array([1 / 2, 1 / 2])

                # Sample a point uniformly from the selected section
                if section == 1:
                    x, y = np.random.multivariate_normal(
                        mean1, np.eye(2) * self.sigma1**2
                    )
                elif section == 2:
                    x, y = np.random.multivariate_normal(
                        mean2, np.eye(2) * self.sigma2**2
                    )
                elif section == 3:
                    x, y = np.random.multivariate_normal(
                        mean3, np.eye(2) * self.sigma3**2
                    )
                elif section == 4:
                    x, y = np.random.multivariate_normal(
                        mean4, np.eye(2) * self.sigma4**2
                    )
                # x, y = np.random.multivariate_normal(origin, np.eye(2)*0.05)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot:
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title("{} samples of the toy dataset with t={}".format(n_samples, t))
            plt.show()

        return samples


class MultiSources_mixture_uni_to_gaussians_v2(data.Dataset):
    """Class for generating the proposed variant of the dataset."""

    def __init__(self, n_samples=100, Max_sources=2, grid_t=False, t=None):
        super(MultiSources_mixture_uni_to_gaussians_v2, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.sigma1 = 0.1
        self.sigma2 = 0.25
        self.sigma3 = 0.05
        self.sigma4 = 0.1

        self.s1 = (-1, 0, -1, 0)
        self.s2 = (-1, 0, 0, 1)
        self.s3 = (0, 1, -1, 0)
        self.s4 = (0, 1, 0, 1)

        self.grid_t = grid_t
        # if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
        # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
        # self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if (
            self.grid_t is True
        ):  # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif (
            self.t is None
        ):  # At training time, the t values are sampled uniformly if the t value is not specificed.
            t = np.random.uniform(0, 1)
        else:
            t = self.t
        mask_activity = np.zeros(
            (self.Max_sources, 1)
        )  # True if the target is active, False otherwise
        mask_activity = mask_activity > 0  # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources, 2))

        # Sample the position of the sources given the number of sources.
        for source in range(N_sources):
            mask_activity[source, 0] = True
            choice_distribution = np.random.choice([1, 2], p=[(1 - t), t])

            s1 = self.s1
            s2 = self.s2
            s3 = self.s3
            s4 = self.s4

            if choice_distribution == 1:

                p1 = p4 = 1 / 2
                p2 = p3 = 0

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
                # Sample a point uniformly from the selected section
                if section == 1:
                    x = np.random.uniform(s1[0], s1[1])
                    y = np.random.uniform(s1[2], s1[3])
                elif section == 2:
                    x = np.random.uniform(s2[0], s2[1])
                    y = np.random.uniform(s2[2], s2[3])
                elif section == 3:
                    x = np.random.uniform(s3[0], s3[1])
                    y = np.random.uniform(s3[2], s3[3])
                elif section == 4:
                    x = np.random.uniform(s4[0], s4[1])
                    y = np.random.uniform(s4[2], s4[3])
                else:
                    x = np.random.uniform(-1, 1)
                    y = np.random.uniform(-1, 1)
                #################

            elif choice_distribution == 2:

                p1 = p4 = 0
                p2 = p3 = 1 / 2

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                mean1 = np.array([-1 / 2, -1 / 2])
                mean2 = np.array([-1 / 2, 1 / 2])
                mean3 = np.array([1 / 2, -1 / 2])
                mean4 = np.array([1 / 2, 1 / 2])

                # Sample a point uniformly from the selected section
                if section == 1:
                    x, y = np.random.multivariate_normal(
                        mean1, np.eye(2) * self.sigma1**2
                    )
                elif section == 2:
                    x, y = np.random.multivariate_normal(
                        mean2, np.eye(2) * self.sigma2**2
                    )
                elif section == 3:
                    x, y = np.random.multivariate_normal(
                        mean3, np.eye(2) * self.sigma3**2
                    )
                elif section == 4:
                    x, y = np.random.multivariate_normal(
                        mean4, np.eye(2) * self.sigma4**2
                    )

            output[source, 0], output[source, 1] = x, y

        return np.array([t]), output, mask_activity

    def define_t_grid(self):
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def generate_dataset_distribution(
        self, t, n_samples, plot_one_sample=False, Max_sources=2
    ):
        """Generate a dataset with a fixed value of t."""
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0  # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources, 2))

            for source in range(N_sources):
                mask_activity[i, source] = True

                ################
                choice_distribution = np.random.choice([1, 2], p=[(1 - t), t])

                s1 = self.s1
                s2 = self.s2
                s3 = self.s3
                s4 = self.s4

                if choice_distribution == 1:
                    # x = np.random.uniform(-1, 1)
                    # y = np.random.uniform(-1, 1)
                    #################
                    p1 = p4 = 1 / 2
                    p2 = p3 = 0

                    section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                    # Sample a point uniformly from the selected section
                    if section == 1:
                        output[source, 0] = np.random.uniform(s1[0], s1[1])
                        output[source, 1] = np.random.uniform(s1[2], s1[3])
                    elif section == 2:
                        output[source, 0] = np.random.uniform(s2[0], s2[1])
                        output[source, 1] = np.random.uniform(s2[2], s2[3])
                    elif section == 3:
                        output[source, 0] = np.random.uniform(s3[0], s3[1])
                        output[source, 1] = np.random.uniform(s3[2], s3[3])
                    elif section == 4:
                        output[source, 0] = np.random.uniform(s4[0], s4[1])
                        output[source, 1] = np.random.uniform(s4[2], s4[3])
                    else:
                        output[source, 0] = np.random.uniform(-1, 1)
                        output[source, 1] = np.random.uniform(-1, 1)
                    #################

                elif choice_distribution == 2:

                    p1 = p4 = 0
                    p2 = p3 = 1 / 2

                    section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                    # origin = np.array([0,0])
                    mean1 = np.array([-1 / 2, -1 / 2])
                    mean2 = np.array([-1 / 2, 1 / 2])
                    mean3 = np.array([1 / 2, -1 / 2])
                    mean4 = np.array([1 / 2, 1 / 2])

                    # Sample a point uniformly from the selected section
                    if section == 1:
                        output[source, 0], output[source, 1] = (
                            np.random.multivariate_normal(
                                mean1, np.eye(2) * self.sigma1**2
                            )
                        )
                    elif section == 2:
                        output[source, 0], output[source, 1] = (
                            np.random.multivariate_normal(
                                mean2, np.eye(2) * self.sigma2**2
                            )
                        )
                    elif section == 3:
                        output[source, 0], output[source, 1] = (
                            np.random.multivariate_normal(
                                mean3, np.eye(2) * self.sigma3**2
                            )
                        )
                    elif section == 4:
                        output[source, 0], output[source, 1] = (
                            np.random.multivariate_normal(
                                mean4, np.eye(2) * self.sigma4**2
                            )
                        )

                ################

                samples[i, source, 0] = output[source, 0]
                samples[i, source, 1] = output[source, 1]

        if plot_one_sample:
            plt.scatter(
                samples[0, :, 0][mask_activity[0, :]],
                samples[0, :, 1][mask_activity[0, :]],
                marker="*",
                c="red",
                s=100,
            )
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title(
                "{} samples of the multi-source toy dataset with t={}".format(1, t)
            )
            plt.show()

        return samples, mask_activity


# Rotating Two moons


class rotating_two_moons(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf.
    """

    def __init__(self, n_samples=10000):
        super(rotating_two_moons, self).__init__()
        self.n_samples = n_samples
        initial_points, _ = datasets.make_moons(n_samples=self.n_samples, noise=0.1)
        self.initial_points = initial_points

        self.initial_points[:, 0] = (
            2
            * (initial_points[:, 0] - (initial_points[:, 0].min()))
            / (initial_points[:, 0].max() - initial_points[:, 0].min())
            - 1
        )
        self.initial_points[:, 1] = (
            2
            * (initial_points[:, 1] - initial_points[:, 1].min())
            / (initial_points[:, 1].max() - initial_points[:, 1].min())
            - 1
        )

    def __len__(self):
        return self.n_samples

    def define_t_grid(self):
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        angle = t * 2 * np.pi

        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        index = np.random.randint(0, self.n_samples)

        vector_sampled = self.initial_points[index, :]
        rotated_sample = np.dot(vector_sampled, rotation_matrix)

        x = rotated_sample[0]
        y = rotated_sample[1]

        return torch.Tensor([t, x, y])

    def generate_dataset_distribution(self, t, n_samples, plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section

        # Generate n_samples samples

        samples, _ = datasets.make_moons(n_samples=n_samples, noise=0.1)
        samples[:, 0] = (
            2
            * (samples[:, 0] - (samples[:, 0].min()))
            / (samples[:, 0].max() - samples[:, 0].min())
            - 1
        )
        samples[:, 1] = (
            2
            * (samples[:, 1] - samples[:, 1].min())
            / (samples[:, 1].max() - samples[:, 1].min())
            - 1
        )

        angle = t * 2 * np.pi
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        rotated_samples = np.dot(samples, rotation_matrix)

        if plot:
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title("{} samples of the toy dataset with t={}".format(n_samples, t))
            plt.show()

        return rotated_samples


class MultiSources_rotating_two_moons(data.Dataset):
    """Class for generating the proposed variant of the dataset."""

    def __init__(self, n_samples=10000, Max_sources=2, grid_t=False, t=None):
        super(MultiSources_rotating_two_moons, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        if (
            self.grid_t is True and self.t is None
        ):  # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

        initial_points, _ = datasets.make_moons(n_samples=self.n_samples, noise=0.1)
        self.initial_points = initial_points

        self.initial_points[:, 0] = (
            2
            * (initial_points[:, 0] - (initial_points[:, 0].min()))
            / (initial_points[:, 0].max() - initial_points[:, 0].min())
            - 1
        )
        self.initial_points[:, 1] = (
            2
            * (initial_points[:, 1] - initial_points[:, 1].min())
            / (initial_points[:, 1].max() - initial_points[:, 1].min())
            - 1
        )

    def __len__(self):
        return self.n_samples

    def define_t_grid(self):
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if (
            self.grid_t is True
        ):  # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif (
            self.t is None
        ):  # At training time, the t values are sampled uniformly if the t value is not specificed.
            t = np.random.uniform(0, 1)
        else:
            t = self.t
        mask_activity = np.zeros(
            (self.Max_sources, 1)
        )  # True if the target is active, False otherwise
        mask_activity = mask_activity > 0  # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources, 2))

        # Sample the position of the sources given the number of sources.
        for source in range(N_sources):
            mask_activity[source, 0] = True
            # Select a section according to the probabilities defined in the paper

            angle = t * 2 * np.pi

            rotation_matrix = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )

            index = np.random.randint(0, self.n_samples)

            vector_sampled = self.initial_points[index, :]
            rotated_sample = np.dot(vector_sampled, rotation_matrix)

            output[source, 0] = rotated_sample[0]
            output[source, 1] = rotated_sample[1]

        return np.array([t]), output, mask_activity

    def generate_dataset_distribution(
        self, t, n_samples, plot_one_sample=False, Max_sources=2
    ):
        """Generate a dataset with a fixed value of t."""
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0  # False everywhere

        initial_points, _ = datasets.make_moons(n_samples=n_samples, noise=0.1)
        initial_points = initial_points

        initial_points[:, 0] = (
            2
            * (initial_points[:, 0] - (initial_points[:, 0].min()))
            / (initial_points[:, 0].max() - initial_points[:, 0].min())
            - 1
        )
        initial_points[:, 1] = (
            2
            * (initial_points[:, 1] - initial_points[:, 1].min())
            / (initial_points[:, 1].max() - initial_points[:, 1].min())
            - 1
        )

        for i in range(n_samples):

            N_sources = 1

            angle = t * 2 * np.pi

            for source in range(N_sources):
                mask_activity[i, source] = (
                    True  # This mask stands for the activity of the target (for handling multiple targets).
                )
                # Select a section according to the probabilities defined in the paper
                ###################

                rotation_matrix = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )

                index = np.random.randint(0, self.n_samples)

                vector_sampled = initial_points[index, :]
                rotated_sample = np.dot(vector_sampled, rotation_matrix)

                samples[i, source, 0] = rotated_sample[0]
                samples[i, source, 1] = rotated_sample[1]
                ###################

        if plot_one_sample:
            plt.scatter(
                samples[0, :, 0][mask_activity[0, :]],
                samples[0, :, 1][mask_activity[0, :]],
                marker="*",
                c="red",
                s=100,
            )
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title(
                "{} samples of the multi-source toy dataset with t={}".format(1, t)
            )
            plt.show()

        return samples, mask_activity


# Changing damier


class changing_damier(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf.
    """

    def __init__(self, n_samples=100):
        super(changing_damier, self).__init__()
        self.n_samples = n_samples

        self.s1 = (-0.5, 0, 0.5, 1)
        self.s2 = (0.5, 1, 0.5, 1)
        self.s3 = (-1, -0.5, 0, 0.5)
        self.s4 = (0, 0.5, 0, 0.5)
        self.s5 = (-0.5, 0, -0.5, 0)
        self.s6 = (0.5, 1, -0.5, 0)
        self.s7 = (-1, -0.5, -1, -0.5)
        self.s8 = (0, 0.5, -1, -0.5)

        self.c1 = (-1, -0.5, 0.5, 1)
        self.c2 = (0, 0.5, 0.5, 1)
        self.c3 = (-0.5, 0, 0, 0.5)
        self.c4 = (0.5, 1, 0, 0.5)
        self.c5 = (-1, -0.5, -0.5, 0)
        self.c6 = (0, 0.5, -0.5, 0)
        self.c7 = (-0.5, 0, -1, -0.5)
        self.c8 = (0.5, 1, -1, -0.5)

    def __len__(self):
        return self.n_samples

    def define_t_grid(self):
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        # Select a section according to the probabilities defined in the paper
        choice_damier = np.random.choice(range(1, 3), p=[1 - t, t])

        section_rows = np.random.choice(range(1, 5), p=[1 / 4] * 4)
        section_cols = np.random.choice(range(1, 3), p=[1 / 2] * 2)

        if choice_damier == 1:

            if section_rows == 1:
                if section_cols == 1:
                    x = np.random.uniform(self.s1[0], self.s1[1])
                    y = np.random.uniform(self.s1[2], self.s1[3])
                elif section_cols == 2:
                    x = np.random.uniform(self.s2[0], self.s2[1])
                    y = np.random.uniform(self.s2[2], self.s2[3])
            elif section_rows == 2:
                if section_cols == 1:
                    x = np.random.uniform(self.s3[0], self.s3[1])
                    y = np.random.uniform(self.s3[2], self.s3[3])
                elif section_cols == 2:
                    x = np.random.uniform(self.s4[0], self.s4[1])
                    y = np.random.uniform(self.s4[2], self.s4[3])
            elif section_rows == 3:
                if section_cols == 1:
                    x = np.random.uniform(self.s5[0], self.s5[1])
                    y = np.random.uniform(self.s5[2], self.s5[3])
                elif section_cols == 2:
                    x = np.random.uniform(self.s6[0], self.s6[1])
                    y = np.random.uniform(self.s6[2], self.s6[3])
            elif section_rows == 4:
                if section_cols == 1:
                    x = np.random.uniform(self.s7[0], self.s7[1])
                    y = np.random.uniform(self.s7[2], self.s7[3])
                elif section_cols == 2:
                    x = np.random.uniform(self.s8[0], self.s8[1])
                    y = np.random.uniform(self.s8[2], self.s8[3])

        elif choice_damier == 2:

            if section_rows == 1:
                if section_cols == 1:
                    x = np.random.uniform(self.c1[0], self.c1[1])
                    y = np.random.uniform(self.c1[2], self.c1[3])
                elif section_cols == 2:
                    x = np.random.uniform(self.c2[0], self.c2[1])
                    y = np.random.uniform(self.c2[2], self.c2[3])
            elif section_rows == 2:
                if section_cols == 1:
                    x = np.random.uniform(self.c3[0], self.c3[1])
                    y = np.random.uniform(self.c3[2], self.c3[3])
                elif section_cols == 2:
                    x = np.random.uniform(self.c4[0], self.c4[1])
                    y = np.random.uniform(self.c4[2], self.c4[3])
            elif section_rows == 3:
                if section_cols == 1:
                    x = np.random.uniform(self.c5[0], self.c5[1])
                    y = np.random.uniform(self.c5[2], self.c5[3])
                elif section_cols == 2:
                    x = np.random.uniform(self.c6[0], self.c6[1])
                    y = np.random.uniform(self.c6[2], self.c6[3])
            elif section_rows == 4:
                if section_cols == 1:
                    x = np.random.uniform(self.c7[0], self.c7[1])
                    y = np.random.uniform(self.c7[2], self.c7[3])
                elif section_cols == 2:
                    x = np.random.uniform(self.c8[0], self.c8[1])
                    y = np.random.uniform(self.c8[2], self.c8[3])

        return torch.Tensor([t, x, y])

    def generate_dataset_distribution(self, t, n_samples, plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""

        samples = np.zeros((n_samples, 2))

        for i in range(n_samples):
            # Select a section
            # Select a section according to the probabilities defined in the paper
            # Select a section according to the probabilities defined in the paper
            choice_damier = np.random.choice(range(1, 3), p=[1 - t, t])

            section_rows = np.random.choice(range(1, 5), p=[1 / 4] * 4)
            section_cols = np.random.choice(range(1, 3), p=[1 / 2] * 2)

            if choice_damier == 1:

                if section_rows == 1:
                    if section_cols == 1:
                        x = np.random.uniform(self.s1[0], self.s1[1])
                        y = np.random.uniform(self.s1[2], self.s1[3])
                    elif section_cols == 2:
                        x = np.random.uniform(self.s2[0], self.s2[1])
                        y = np.random.uniform(self.s2[2], self.s2[3])
                elif section_rows == 2:
                    if section_cols == 1:
                        x = np.random.uniform(self.s3[0], self.s3[1])
                        y = np.random.uniform(self.s3[2], self.s3[3])
                    elif section_cols == 2:
                        x = np.random.uniform(self.s4[0], self.s4[1])
                        y = np.random.uniform(self.s4[2], self.s4[3])
                elif section_rows == 3:
                    if section_cols == 1:
                        x = np.random.uniform(self.s5[0], self.s5[1])
                        y = np.random.uniform(self.s5[2], self.s5[3])
                    elif section_cols == 2:
                        x = np.random.uniform(self.s6[0], self.s6[1])
                        y = np.random.uniform(self.s6[2], self.s6[3])
                elif section_rows == 4:
                    if section_cols == 1:
                        x = np.random.uniform(self.s7[0], self.s7[1])
                        y = np.random.uniform(self.s7[2], self.s7[3])
                    elif section_cols == 2:
                        x = np.random.uniform(self.s8[0], self.s8[1])
                        y = np.random.uniform(self.s8[2], self.s8[3])

            elif choice_damier == 2:

                if section_rows == 1:
                    if section_cols == 1:
                        x = np.random.uniform(self.c1[0], self.c1[1])
                        y = np.random.uniform(self.c1[2], self.c1[3])
                    elif section_cols == 2:
                        x = np.random.uniform(self.c2[0], self.c2[1])
                        y = np.random.uniform(self.c2[2], self.c2[3])
                elif section_rows == 2:
                    if section_cols == 1:
                        x = np.random.uniform(self.c3[0], self.c3[1])
                        y = np.random.uniform(self.c3[2], self.c3[3])
                    elif section_cols == 2:
                        x = np.random.uniform(self.c4[0], self.c4[1])
                        y = np.random.uniform(self.c4[2], self.c4[3])
                elif section_rows == 3:
                    if section_cols == 1:
                        x = np.random.uniform(self.c5[0], self.c5[1])
                        y = np.random.uniform(self.c5[2], self.c5[3])
                    elif section_cols == 2:
                        x = np.random.uniform(self.c6[0], self.c6[1])
                        y = np.random.uniform(self.c6[2], self.c6[3])
                elif section_rows == 4:
                    if section_cols == 1:
                        x = np.random.uniform(self.c7[0], self.c7[1])
                        y = np.random.uniform(self.c7[2], self.c7[3])
                    elif section_cols == 2:
                        x = np.random.uniform(self.c8[0], self.c8[1])
                        y = np.random.uniform(self.c8[2], self.c8[3])

            samples[i, 0] = x
            samples[i, 1] = y

        if plot:
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title("{} samples of the toy dataset with t={}".format(n_samples, t))
            plt.show()

        return samples


class MultiSources_changing_damier(data.Dataset):
    """Class for generating the proposed variant of the dataset."""

    def __init__(self, n_samples=100, Max_sources=2, grid_t=False, t=None):
        super(MultiSources_changing_damier, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        if (
            self.grid_t is True and self.t is None
        ):  # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

        self.s1 = (-0.5, 0, 0.5, 1)
        self.s2 = (0.5, 1, 0.5, 1)
        self.s3 = (-1, -0.5, 0, 0.5)
        self.s4 = (0, 0.5, 0, 0.5)
        self.s5 = (-0.5, 0, -0.5, 0)
        self.s6 = (0.5, 1, -0.5, 0)
        self.s7 = (-1, -0.5, -1, -0.5)
        self.s8 = (0, 0.5, -1, -0.5)

        self.c1 = (-1, -0.5, 0.5, 1)
        self.c2 = (0, 0.5, 0.5, 1)
        self.c3 = (-0.5, 0, 0, 0.5)
        self.c4 = (0.5, 1, 0, 0.5)
        self.c5 = (-1, -0.5, -0.5, 0)
        self.c6 = (0, 0.5, -0.5, 0)
        self.c7 = (-0.5, 0, -1, -0.5)
        self.c8 = (0.5, 1, -1, -0.5)

    def __len__(self):
        return self.n_samples

    def define_t_grid(self):
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if (
            self.grid_t is True
        ):  # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif (
            self.t is None
        ):  # At training time, the t values are sampled uniformly if the t value is not specificed.
            t = np.random.uniform(0, 1)
        else:
            t = self.t
        mask_activity = np.zeros(
            (self.Max_sources, 1)
        )  # True if the target is active, False otherwise
        mask_activity = mask_activity > 0  # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources, 2))

        # Sample the position of the sources given the number of sources.
        for source in range(N_sources):
            mask_activity[source, 0] = True

            ##################
            choice_damier = np.random.choice(range(1, 3), p=[1 - t, t])

            section_rows = np.random.choice(range(1, 5), p=[1 / 4] * 4)
            section_cols = np.random.choice(range(1, 3), p=[1 / 2] * 2)

            if choice_damier == 1:

                if section_rows == 1:
                    if section_cols == 1:
                        output[source, 0] = np.random.uniform(self.s1[0], self.s1[1])
                        output[source, 1] = np.random.uniform(self.s1[2], self.s1[3])
                    elif section_cols == 2:
                        output[source, 0] = np.random.uniform(self.s2[0], self.s2[1])
                        output[source, 1] = np.random.uniform(self.s2[2], self.s2[3])
                elif section_rows == 2:
                    if section_cols == 1:
                        output[source, 0] = np.random.uniform(self.s3[0], self.s3[1])
                        output[source, 1] = np.random.uniform(self.s3[2], self.s3[3])
                    elif section_cols == 2:
                        output[source, 0] = np.random.uniform(self.s4[0], self.s4[1])
                        output[source, 1] = np.random.uniform(self.s4[2], self.s4[3])
                elif section_rows == 3:
                    if section_cols == 1:
                        output[source, 0] = np.random.uniform(self.s5[0], self.s5[1])
                        output[source, 1] = np.random.uniform(self.s5[2], self.s5[3])
                    elif section_cols == 2:
                        output[source, 0] = np.random.uniform(self.s6[0], self.s6[1])
                        output[source, 1] = np.random.uniform(self.s6[2], self.s6[3])
                elif section_rows == 4:
                    if section_cols == 1:
                        output[source, 0] = np.random.uniform(self.s7[0], self.s7[1])
                        output[source, 1] = np.random.uniform(self.s7[2], self.s7[3])
                    elif section_cols == 2:
                        output[source, 0] = np.random.uniform(self.s8[0], self.s8[1])
                        output[source, 1] = np.random.uniform(self.s8[2], self.s8[3])

            elif choice_damier == 2:

                if section_rows == 1:
                    if section_cols == 1:
                        output[source, 0] = np.random.uniform(self.c1[0], self.c1[1])
                        output[source, 1] = np.random.uniform(self.c1[2], self.c1[3])
                    elif section_cols == 2:
                        output[source, 0] = np.random.uniform(self.c2[0], self.c2[1])
                        output[source, 1] = np.random.uniform(self.c2[2], self.c2[3])
                elif section_rows == 2:
                    if section_cols == 1:
                        output[source, 0] = np.random.uniform(self.c3[0], self.c3[1])
                        output[source, 1] = np.random.uniform(self.c3[2], self.c3[3])
                    elif section_cols == 2:
                        output[source, 0] = np.random.uniform(self.c4[0], self.c4[1])
                        output[source, 1] = np.random.uniform(self.c4[2], self.c4[3])
                elif section_rows == 3:
                    if section_cols == 1:
                        output[source, 0] = np.random.uniform(self.c5[0], self.c5[1])
                        output[source, 1] = np.random.uniform(self.c5[2], self.c5[3])
                    elif section_cols == 2:
                        output[source, 0] = np.random.uniform(self.c6[0], self.c6[1])
                        output[source, 1] = np.random.uniform(self.c6[2], self.c6[3])
                elif section_rows == 4:
                    if section_cols == 1:
                        output[source, 0] = np.random.uniform(self.c7[0], self.c7[1])
                        output[source, 1] = np.random.uniform(self.c7[2], self.c7[3])
                    elif section_cols == 2:
                        output[source, 0] = np.random.uniform(self.c8[0], self.c8[1])
                        output[source, 1] = np.random.uniform(self.c8[2], self.c8[3])

        return np.array([t]), output, mask_activity

    def generate_dataset_distribution(
        self, t, n_samples, plot_one_sample=False, Max_sources=2
    ):
        """Generate a dataset with a fixed value of t."""
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0  # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources, 2))

            for source in range(N_sources):
                mask_activity[i, source] = (
                    True  # This mask stands for the activity of the target (for handling multiple targets).
                )
                # Select a section according to the probabilities defined in the paper
                ####################
                ##################
                choice_damier = np.random.choice(range(1, 3), p=[1 - t, t])

                section_rows = np.random.choice(range(1, 5), p=[1 / 4] * 4)
                section_cols = np.random.choice(range(1, 3), p=[1 / 2] * 2)

                if choice_damier == 1:

                    if section_rows == 1:
                        if section_cols == 1:
                            output[source, 0] = np.random.uniform(
                                self.s1[0], self.s1[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.s1[2], self.s1[3]
                            )
                        elif section_cols == 2:
                            output[source, 0] = np.random.uniform(
                                self.s2[0], self.s2[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.s2[2], self.s2[3]
                            )
                    elif section_rows == 2:
                        if section_cols == 1:
                            output[source, 0] = np.random.uniform(
                                self.s3[0], self.s3[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.s3[2], self.s3[3]
                            )
                        elif section_cols == 2:
                            output[source, 0] = np.random.uniform(
                                self.s4[0], self.s4[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.s4[2], self.s4[3]
                            )
                    elif section_rows == 3:
                        if section_cols == 1:
                            output[source, 0] = np.random.uniform(
                                self.s5[0], self.s5[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.s5[2], self.s5[3]
                            )
                        elif section_cols == 2:
                            output[source, 0] = np.random.uniform(
                                self.s6[0], self.s6[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.s6[2], self.s6[3]
                            )
                    elif section_rows == 4:
                        if section_cols == 1:
                            output[source, 0] = np.random.uniform(
                                self.s7[0], self.s7[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.s7[2], self.s7[3]
                            )
                        elif section_cols == 2:
                            output[source, 0] = np.random.uniform(
                                self.s8[0], self.s8[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.s8[2], self.s8[3]
                            )

                elif choice_damier == 2:

                    if section_rows == 1:
                        if section_cols == 1:
                            output[source, 0] = np.random.uniform(
                                self.c1[0], self.c1[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.c1[2], self.c1[3]
                            )
                        elif section_cols == 2:
                            output[source, 0] = np.random.uniform(
                                self.c2[0], self.c2[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.c2[2], self.c2[3]
                            )
                    elif section_rows == 2:
                        if section_cols == 1:
                            output[source, 0] = np.random.uniform(
                                self.c3[0], self.c3[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.c3[2], self.c3[3]
                            )
                        elif section_cols == 2:
                            output[source, 0] = np.random.uniform(
                                self.c4[0], self.c4[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.c4[2], self.c4[3]
                            )
                    elif section_rows == 3:
                        if section_cols == 1:
                            output[source, 0] = np.random.uniform(
                                self.c5[0], self.c5[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.c5[2], self.c5[3]
                            )
                        elif section_cols == 2:
                            output[source, 0] = np.random.uniform(
                                self.c6[0], self.c6[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.c6[2], self.c6[3]
                            )
                    elif section_rows == 4:
                        if section_cols == 1:
                            output[source, 0] = np.random.uniform(
                                self.c7[0], self.c7[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.c7[2], self.c7[3]
                            )
                        elif section_cols == 2:
                            output[source, 0] = np.random.uniform(
                                self.c8[0], self.c8[1]
                            )
                            output[source, 1] = np.random.uniform(
                                self.c8[2], self.c8[3]
                            )
                ####################

                samples[i, source, 0] = output[source, 0]
                samples[i, source, 1] = output[source, 1]

        if plot_one_sample:
            plt.scatter(
                samples[0, :, 0][mask_activity[0, :]],
                samples[0, :, 1][mask_activity[0, :]],
                marker="*",
                c="red",
                s=100,
            )
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title(
                "{} samples of the multi-source toy dataset with t={}".format(1, t)
            )
            plt.show()

        return samples, mask_activity


# Single Gaussian not centered


class single_gauss_not_centered(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf.
    """

    def __init__(self, n_samples=100):
        super(single_gauss_not_centered, self).__init__()
        self.n_samples = n_samples
        self.sigma1 = 0.2
        self.mean1 = np.array([0.25, 0.25])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

        # Sample a point uniformly from the selected section
        x, y = np.random.multivariate_normal(self.mean1, np.eye(2) * self.sigma1**2)

        return torch.Tensor([t, x, y])

    def generate_dataset_distribution(self, t, n_samples, plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""

        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            x, y = np.random.multivariate_normal(self.mean1, np.eye(2) * self.sigma1**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot:
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title("{} samples of the toy dataset with t={}".format(n_samples, t))
            plt.show()

        return samples


class MultiSources_single_gauss_not_centered(data.Dataset):
    """Class for generating the proposed variant of the dataset."""

    def __init__(self, n_samples=100, Max_sources=2, grid_t=False, t=None):
        super(MultiSources_single_gauss_not_centered, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.sigma1 = 0.2
        self.mean1 = np.array([0.25, 0.25])

        self.grid_t = grid_t

    def __len__(self):
        return self.n_samples

    def define_t_grid(self):
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if (
            self.grid_t is True
        ):  # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif (
            self.t is None
        ):  # At training time, the t values are sampled uniformly if the t value is not specificed.
            t = np.random.uniform(0, 1)
        else:
            t = self.t
        mask_activity = np.zeros(
            (self.Max_sources, 1)
        )  # True if the target is active, False otherwise
        mask_activity = mask_activity > 0  # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources, 2))

        # Sample the position of the sources given the number of sources.
        for source in range(N_sources):
            mask_activity[source, 0] = True
            # Select a section according to the probabilities defined in the paper

            choice_distribution = np.random.choice([1, 2], p=[(1 - t), t])

            x, y = np.random.multivariate_normal(self.mean1, np.eye(2) * self.sigma1**2)

            output[source, 0], output[source, 1] = x, y

        return np.array([t]), output, mask_activity

    def generate_dataset_distribution(
        self, t, n_samples, plot_one_sample=False, Max_sources=2
    ):
        """Generate a dataset with a fixed value of t."""
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0  # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources, 2))

            for source in range(N_sources):
                mask_activity[i, source] = (
                    True  # This mask stands for the activity of the target (for handling multiple targets).
                )

                x, y = np.random.multivariate_normal(
                    self.mean1, np.eye(2) * self.sigma1**2
                )

                output[source, 0], output[source, 1] = x, y

                samples[i, source, 0] = output[source, 0]
                samples[i, source, 1] = output[source, 1]

        if plot_one_sample:
            plt.scatter(
                samples[0, :, 0][mask_activity[0, :]],
                samples[0, :, 1][mask_activity[0, :]],
                marker="*",
                c="red",
                s=100,
            )
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title(
                "{} samples of the multi-source toy dataset with t={}".format(1, t)
            )
            plt.show()

        return samples, mask_activity
